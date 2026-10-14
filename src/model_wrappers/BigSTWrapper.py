import math
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def _random_projection(num_features: int, dim: int, seed: int) -> torch.Tensor:
    """Orthogonal random features (Performer): blocks of random orthogonal rows, scaled by the norms of
    Gaussian rows; [num_features, dim]. Drawn from a local generator, so that the global random state
    (dropout, data shuffling) is left untouched, unlike the official code (torch.manual_seed)."""
    generator = torch.Generator().manual_seed(seed)
    blocks = []
    for _ in range(math.ceil(num_features / dim)):
        q, _ = torch.linalg.qr(torch.randn((dim, dim), generator=generator))
        blocks.append(q.T)
    matrix = torch.vstack(blocks)[:num_features]
    multiplier = torch.norm(torch.randn((num_features, dim), generator=generator), dim=1)
    return multiplier.unsqueeze(1) * matrix


def _random_feature_map(data: torch.Tensor, is_query: bool, projection: torch.Tensor, eps: float = 1e-6):
    """Positive random features of the softmax kernel, exp(q·k) ≈ φ(q)·φ(k): [B, N, 1, d] → [B, N, 1, r]."""
    data = data * data.shape[-1] ** -0.25
    ratio = projection.shape[0] ** -0.5
    data_dash = torch.einsum('bnhd,md->bnhm', data, projection)
    diag = (data ** 2).sum(dim=-1, keepdim=True) / 2.0
    if is_query:
        stabiliser = data_dash.max(dim=-1, keepdim=True)[0]
    else:   # one stabiliser for all the keys (nodes)
        stabiliser = data_dash.max(dim=-1, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    return ratio * (torch.exp(data_dash - diag - stabiliser) + eps)


def _linear_kernel(x: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Softmax-kernel graph convolution in O(N): out_i = Σ_j φ(q_i)·φ(k_j) x_j / Σ_j φ(q_i)·φ(k_j).
    x: [B, N, 1, C], query / key: [B, N, 1, r]."""
    kx = torch.einsum('bnhm,bnhd->bhmd', key, x)
    numerator = torch.einsum('bnhm,bhmd->bnhd', query, kx)
    denominator = torch.einsum('bnhm,bhm->bnh', query, key.sum(dim=1)).unsqueeze(-1)
    return numerator / denominator


def _spatial_loss(query, key, edge_index, edge_weight):
    """Official spatial regulariser: weighted log-attention of the edges of a given graph (here the
    operational-context graph), so that the learned (approximated) attention follows it.
    edge_index [2, E] (source, target): the target node i attends to the source node j."""
    target, source = edge_index[1], edge_index[0]
    attention = (query[:, target] * key[:, source]).sum(dim=-1)                       # [B, E, 1]
    normaliser = torch.einsum('bnhm,bhm->bnh', query, key.sum(dim=1))[:, target]       # [B, E, 1]
    return torch.mean((attention / normaliser).log() * edge_weight.reshape(1, -1, 1))


class LinearizedConv(nn.Module):
    """Gated projection of the node representations, then the linearised (softmax-kernel) graph convolution
    with node vectors from the node embeddings."""

    def __init__(self, dim: int, dropout: float, tau: float, random_feature_dim: int):
        super().__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.input_fc = nn.Conv2d(dim, dim, kernel_size=(1, 1))
        self.output_fc = nn.Conv2d(dim, dim, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_vec1, node_vec2):
        # x: [B, C, N, 1], node_vec1 / node_vec2: [B, N, 1, d]
        x = self.dropout(torch.sigmoid(self.input_fc(x)) * self.output_fc(x))
        # As the official code: features redrawn at every call, seeded by the node vectors (so fixed once trained)
        seed = int(torch.ceil(torch.abs(node_vec1.detach().sum()) * 1e8).item()) % (2 ** 31)
        projection = _random_projection(self.random_feature_dim, node_vec1.shape[-1], seed).to(x.device)
        query = _random_feature_map(node_vec1 / math.sqrt(self.tau), True, projection)
        key = _random_feature_map(node_vec2 / math.sqrt(self.tau), False, projection)
        out = _linear_kernel(x.permute(0, 2, 3, 1), query, key)                          # [B, N, 1, C]
        return out.permute(0, 3, 1, 2), query, key                                        # [B, C, N, 1]


class BigSTModel(nn.Module):
    """
    BigST: Linear Complexity Spatio-Temporal Graph Neural Network, adapted to next-step forecasting.

    The window of every node is flattened and embedded (1×1 convolution of its T·F values) and concatenated
    with a learnable node embedding. Stacked linearised spatial convolutions follow: a graph convolution
    whose softmax attention between nodes (from two projections of the node embeddings) is approximated by
    random features, linear in the number of nodes N; residual connections and layer norm. A regression
    layer on the input and the last layer gives the forecast.

    Follows the official implementation (github.com/usail-hkust/BigST, model.py) with its default options
    (--use_residual --use_bn), without the time-of-day / day-of-week embeddings (not used for the other
    models here) and without the long-history features (use_long: preprocessed histories of weeks, the
    windows here are a few steps). The spatial regulariser (use_spatial, off by default as in the
    official run) is optional and takes the operational-context graph.

    Reference: Han et al., "BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic
               Forecasting on Large-Scale Road Networks", PVLDB 17(5), 2024.
    """

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 hid_dim: int = 32, node_dim: int = 32, num_layers: int = 3, tau: float = 0.25,
                 random_feature_dim: int = 64, dropout: float = 0.3, out_features: int = None):
        super().__init__()
        out_features = out_features or node_features
        dim = hid_dim + node_dim
        self.node_emb = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_nodes, node_dim)))
        self.input_emb = nn.Conv2d(slide_win * node_features, hid_dim, kernel_size=(1, 1))
        self.W_1 = nn.Conv2d(node_dim, hid_dim, kernel_size=(1, 1))
        self.W_2 = nn.Conv2d(node_dim, hid_dim, kernel_size=(1, 1))
        self.layers = nn.ModuleList([LinearizedConv(dim, dropout, tau, random_feature_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.regression = nn.Conv2d(2 * dim, out_features, kernel_size=(1, 1))
        self.query = self.key = None   # random features of the last layer, for the spatial regulariser

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F]
        B, T, N, F_in = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, T * F_in, N, 1)                             # [B, T·F, N, 1]
        node_emb = self.node_emb.T.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)      # [B, d, N, 1]
        x = torch.cat([self.input_emb(x), node_emb], dim=1)                              # [B, C, N, 1]
        node_vec1 = self.W_1(node_emb).permute(0, 2, 3, 1)                               # [B, N, 1, hid]
        node_vec2 = self.W_2(node_emb).permute(0, 2, 3, 1)
        x_pool = [x]
        for layer, norm in zip(self.layers, self.norms):
            h, self.query, self.key = layer(x, node_vec1, node_vec2)
            x = norm((h + x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = torch.relu(torch.cat(x_pool + [x], dim=1))                                   # [B, 2C, N, 1]
        return self.regression(x).squeeze(-1).transpose(1, 2)                           # [B, N, F_out]


class BigSTWrapper:

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 hid_dim: int = 32, node_dim: int = 32, num_layers: int = 3, tau: float = 0.25,
                 random_feature_dim: int = 64, dropout: float = 0.3,
                 use_spatial: bool = False, spatial_weight: float = 0.3, edge_index=None, edge_weight=None,
                 null_padding_feature: bool = False, null_padding_target: bool = False,
                 lr: float = 1e-3, weight_decay: float = 1e-4, clip_grad: float = 5.0,
                 batch_size: int = 32, device='cpu'):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.slide_win = slide_win
        self.hid_dim = hid_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.dropout = dropout
        # Spatial regulariser on the operational-context graph (edge_index [2, E], edge_weight [E])
        self.use_spatial = use_spatial
        self.spatial_weight = spatial_weight
        if use_spatial:
            assert edge_index is not None and edge_weight is not None, 'use_spatial needs the graph edges'
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad   # max gradient norm; 0 / None: no clipping
        # Null padding appends an is_nan channel to the inputs and/or targets
        self.null_padding_feature = null_padding_feature
        self.null_padding_target = null_padding_target
        self.batch_size = batch_size
        self.device = device
        self.inference_time = 0
        self._init_model()

    def _init_model(self):
        self.model = BigSTModel(
            num_nodes=self.num_nodes,
            node_features=self.node_features + int(self.null_padding_feature),
            slide_win=self.slide_win,
            hid_dim=self.hid_dim,
            node_dim=self.node_dim,
            num_layers=self.num_layers,
            tau=self.tau,
            random_feature_dim=self.random_feature_dim,
            dropout=self.dropout,
            out_features=self.node_features + int(self.null_padding_target),
        ).to(self.device)
        # Official optimizer (its learning-rate milestones, epochs 80 and 100, are past the epochs run here)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-8)
        self.loss_fn = nn.MSELoss()
        print(f'BigSTWrapper — device: {self.device}, nodes: {self.num_nodes}, '
              f'features: {self.node_features}, hidden: {self.hid_dim} + node {self.node_dim}, '
              f'layers: {self.num_layers}, random features: {self.random_feature_dim}, '
              f'spatial loss: {self.use_spatial}, lr: {self.lr}, weight decay: {self.weight_decay}, '
              f'null padding feature/target: {self.null_padding_feature}/{self.null_padding_target}  '
              f'[{sum(p.numel() for p in self.model.parameters())} params]')

    def train_step(self, inputs, labels) -> float:
        y_hat = self.model(inputs)
        loss = self.loss_fn(y_hat, labels)
        objective = loss
        if self.use_spatial:   # as the official trainer: loss - 0.3 · spatial log-likelihood of the graph
            objective = loss - self.spatial_weight * _spatial_loss(self.model.query, self.model.key,
                                                                   self.edge_index, self.edge_weight)
        self.optimizer.zero_grad()
        objective.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        return loss.item()

    def train(self, train_loader, val_loader, epochs: int):
        train_losses, valid_losses = [], []
        t0 = time.time()

        for epoch in range(epochs):
            self.model.train()
            loss_list = [self.train_step(inputs, labels)
                         for inputs, labels in tqdm(train_loader, total=len(train_loader), desc='Training...')]

            epoch_train_loss = sum(loss_list) / len(loss_list)
            train_losses.append(epoch_train_loss)
            valid_predictions, _, _, epoch_valid_loss = self.predict(val_loader, mode='valid')
            valid_losses.append(epoch_valid_loss)
            print(f'Epoch {epoch} train MSE: {epoch_train_loss:.7f}, valid MSE: {epoch_valid_loss:.7f}')
        self._warn_if_collapsed(valid_predictions)

        self.history = {
            'epochs': epochs,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'training_time': time.time() - t0,
        }
        return self.history

    @staticmethod
    def _warn_if_collapsed(predictions, tolerance=1e-6):
        """Warns when the predictions [T, N, F] barely change over time: the model ignores its inputs."""
        spread = float(np.median(predictions.std(axis=0)))
        if spread < tolerance:
            print(f'WARNING: BigST predictions are almost constant over time (median std per sensor '
                  f'{spread:.2e} < {tolerance:.0e}): the model has collapsed and ignores its inputs; '
                  f'its anomaly scores are meaningless')

    def predict(self, loader, mode: str):
        self.model.eval()
        total_loss, errors, preds = [], [], []
        is_nan_preds, is_nan_labels = [], []
        F_data = self.node_features
        t0 = time.time()

        with torch.no_grad():
            for inputs, labels in tqdm(loader, total=len(loader), desc='Testing...'):
                y_hat = self.model(inputs)
                total_loss.append(self.loss_fn(y_hat, labels).item())
                # Score only the data channels; the is_nan channel is tracked separately
                preds.append(y_hat[:, :, :F_data].cpu().numpy())
                errors.append((y_hat - labels)[:, :, :F_data].abs().cpu().numpy())
                if self.null_padding_target:
                    is_nan_preds.append(y_hat[:, :, -1].cpu().numpy())
                    is_nan_labels.append(labels[:, :, -1].cpu().numpy())

        if mode == 'test':
            self.inference_time = time.time() - t0

        is_nan_results = None
        if self.null_padding_target:
            is_nan_results = np.array([np.concatenate(is_nan_preds, axis=0),
                                       np.concatenate(is_nan_labels, axis=0)])   # [2, total, N]

        return (
            np.concatenate(preds, axis=0),          # [total, N, F]
            is_nan_results,
            np.concatenate(errors, axis=0),         # [total, N, F]
            sum(total_loss) / len(total_loss),
        )

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f'Model loaded from {path}')
