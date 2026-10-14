import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no centring), as the official code."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) * x.size(-1) ** -0.5
        return self.scale * x / (rms + self.eps)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, expand_ratio: int = 4, dropout: float = 0.3):
        super().__init__()
        self.W1 = nn.Linear(dim, expand_ratio * dim)
        self.W2 = nn.Linear(dim, expand_ratio * dim)
        self.W3 = nn.Linear(expand_ratio * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))


class MAGEBlock(nn.Module):
    """
    Mixture of adaptive graph experts: K experts, each a low-rank graph A_k = Softmax(E1_k) Softmax(E2_k)
    (two copies combined differentially, A = A⁽¹⁾ - λ A⁽²⁾); every node is routed to its top-k experts.
    The graphs are never materialised: the propagation is computed as Softmax(E1) (Softmax(E2) x), linear
    in the number of nodes N.
    """

    def __init__(self, dim: int, num_nodes: int, num_experts: int, graph_dim: int, topk: int, depth: int):
        super().__init__()
        self.topk = topk
        self.E1 = nn.Parameter(torch.rand(2, num_experts, num_nodes, graph_dim))
        self.E2 = nn.Parameter(torch.rand(2, num_experts, graph_dim, num_nodes))
        self.router = nn.Linear(dim, num_experts)
        # Differential combination of the two graph copies (as differential attention): coefficients [1, λ]
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.empty(dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.empty(dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.empty(dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.empty(dim).normal_(mean=0, std=0.1))
        # The official output map (cal_outer) mixes an identity with itself: always the identity, left out

    def coefficients(self) -> torch.Tensor:
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        return torch.stack([torch.ones_like(lambda_1), -(lambda_1 - lambda_2 + self.lambda_init)])   # [2]

    def route(self, x: torch.Tensor, bias: torch.Tensor):
        """Sparse routing: weights of the top-k experts of every node (normalised to sum 1), and their indices.
        The balancing bias only changes which experts are selected, not their weights."""
        logits = self.router(x)                                                      # [B, N, K]
        _, topk_indices = torch.topk(torch.sigmoid(logits) + bias, self.topk, dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, topk_indices, True)
        weights = torch.sigmoid(logits) * mask
        return weights / weights.sum(dim=-1, keepdim=True), topk_indices

    def forward(self, x: torch.Tensor, bias: torch.Tensor):
        # x: [B, N, D]
        B, N, D = x.shape
        weights, topk_indices = self.route(x, bias)                                  # [B, N, K]
        E1 = torch.softmax(self.E1, dim=-1)                                          # [2, K, N, d]
        E2 = torch.softmax(self.E2, dim=-1)                                          # [2, K, d, N]
        # Same result as the official einsum('bik, e, ekid, ekdj, bjf -> bif'), contracted in an order
        # linear in N (the N×N graphs are never formed)
        y = torch.einsum('ekdj,bjf->bekdf', E2, x)                                   # [B, 2, K, d, D]
        w = torch.einsum('bik,e,ekid->biekd', weights, self.coefficients(), E1)      # [B, N, 2, K, d]
        out = torch.bmm(w.reshape(B, N, -1), y.reshape(B, -1, D))                    # [B, N, D]
        return out, topk_indices


class PreNormBlock(nn.Module):
    def __init__(self, dim, num_nodes, num_experts, graph_dim, topk, depth, dropout):
        super().__init__()
        self.graph = MAGEBlock(dim, num_nodes, num_experts, graph_dim, topk, depth)
        self.graph_norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, dropout=dropout)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, bias):
        h, topk_indices = self.graph(self.graph_norm(x), bias)
        x = x + h
        return x + self.ffn(self.ffn_norm(x)), topk_indices


class MAGEModel(nn.Module):
    """
    MAGE: Mixture of Adaptive Graph Experts, adapted to next-step forecasting.

    The window of every node is flattened and embedded (linear map of its T·F values) and a learnable
    spatial prompt (node embedding) is added. Three pre-norm blocks follow, each a graph propagation by a
    sparse mixture of adaptive graph experts and a SwiGLU feed-forward; the third block takes the residual
    of the second, and two linear heads (after the first and the third block) are summed.

    Follows the official implementation (github.com/PoorOtterBob/MAGE, models/MAGE.py) without the
    temporal prompts (time-of-day / day-of-week embeddings, not used for the other models here) and with
    a single forecasting step.

    Reference: Ma et al., "Less but More: Linear Adaptive Graph Learning Empowering Spatiotemporal
               Forecasting", NeurIPS 2025.
    """

    NUM_BLOCKS = 3

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 model_dim: int = 128, num_experts: int = 16, graph_dim: int = 32, topk: int = 4,
                 dropout: float = 0.3, out_features: int = None):
        super().__init__()
        assert 1 <= topk <= num_experts, f'topk must be in [1, num_experts={num_experts}], got {topk}'
        out_features = out_features or node_features
        self.num_experts = num_experts
        self.encoder = nn.Linear(slide_win * node_features, model_dim)
        self.spatial_prompt = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_nodes, model_dim)))
        self.blocks = nn.ModuleList([PreNormBlock(model_dim, num_nodes, num_experts, graph_dim, topk, depth, dropout)
                                     for depth in range(1, self.NUM_BLOCKS + 1)])
        self.decoder = nn.Linear(model_dim, out_features)
        self.decoder2 = nn.Linear(model_dim, out_features)
        # Load-balancing bias of the experts of every block, updated outside back-propagation
        # (update_balance) so that every expert is selected about as often
        self.register_buffer('expert_bias', torch.zeros(self.NUM_BLOCKS, num_experts))
        self.expert_load = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F]
        B, T, N, F_in = x.shape
        h = self.encoder(x.transpose(1, 2).reshape(B, N, T * F_in)) + self.spatial_prompt   # [B, N, D]
        z, idx1 = self.blocks[0](h, self.expert_bias[0])
        x, idx2 = self.blocks[1](z, self.expert_bias[1])
        x, idx3 = self.blocks[2](h - x, self.expert_bias[2])
        if self.training:
            self.expert_load = torch.stack([torch.bincount(i.flatten(), minlength=self.num_experts)
                                            for i in (idx1, idx2, idx3)])                  # [3, K]
        return self.decoder(x) + self.decoder2(z)                                          # [B, N, F_out]

    @torch.no_grad()
    def update_balance(self, lr: float):
        """Auxiliary-loss-free balancing: lower the bias of the experts selected more than on average."""
        if self.expert_load is None or lr == 0:
            return
        load = self.expert_load.float()
        self.expert_bias -= lr * torch.sign(load - load.mean(dim=-1, keepdim=True))


class MAGEWrapper:

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 model_dim: int = 128, num_experts: int = 16, graph_dim: int = 32, topk: int = 4,
                 dropout: float = 0.3, balance_lr: float = 1e-3,
                 null_padding_feature: bool = False, null_padding_target: bool = False,
                 lr: float = 1e-2, weight_decay: float = 1e-4, lr_milestones=(1, 30, 50, 70, 90),
                 clip_grad: float = 5.0, batch_size: int = 32, device='cpu'):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.slide_win = slide_win
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.graph_dim = graph_dim
        self.topk = topk
        self.dropout = dropout
        self.balance_lr = balance_lr   # step of the expert-balancing bias; 0: no balancing
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_milestones = list(lr_milestones)   # epochs halving the learning rate, as the official schedule
        self.clip_grad = clip_grad   # max gradient norm; 0 / None: no clipping
        # Null padding appends an is_nan channel to the inputs and/or targets
        self.null_padding_feature = null_padding_feature
        self.null_padding_target = null_padding_target
        self.batch_size = batch_size
        self.device = device
        self.inference_time = 0
        self._init_model()

    def _init_model(self):
        self.model = MAGEModel(
            num_nodes=self.num_nodes,
            node_features=self.node_features + int(self.null_padding_feature),
            slide_win=self.slide_win,
            model_dim=self.model_dim,
            num_experts=self.num_experts,
            graph_dim=self.graph_dim,
            topk=self.topk,
            dropout=self.dropout,
            out_features=self.node_features + int(self.null_padding_target),
        ).to(self.device)
        # Official optimizer and schedule: Adam with L2 weight decay, learning rate halved at the milestones
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_milestones, gamma=0.5)
        self.loss_fn = nn.MSELoss()
        print(f'MAGEWrapper — device: {self.device}, nodes: {self.num_nodes}, '
              f'features: {self.node_features}, model_dim: {self.model_dim}, '
              f'experts: {self.num_experts} (top-{self.topk}, graph dim {self.graph_dim}), '
              f'lr: {self.lr}, weight decay: {self.weight_decay}, clip grad: {self.clip_grad}, '
              f'null padding feature/target: {self.null_padding_feature}/{self.null_padding_target}  '
              f'[{sum(p.numel() for p in self.model.parameters())} params]')

    def train_step(self, inputs, labels) -> float:
        y_hat = self.model(inputs)
        loss = self.loss_fn(y_hat, labels)
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        self.model.update_balance(self.balance_lr)
        return loss.item()

    def train(self, train_loader, val_loader, epochs: int):
        train_losses, valid_losses = [], []
        t0 = time.time()

        for epoch in range(epochs):
            self.model.train()
            loss_list = [self.train_step(inputs, labels)
                         for inputs, labels in tqdm(train_loader, total=len(train_loader), desc='Training...')]
            self.scheduler.step()

            epoch_train_loss = sum(loss_list) / len(loss_list)
            train_losses.append(epoch_train_loss)
            valid_predictions, _, _, epoch_valid_loss = self.predict(val_loader, mode='valid')
            valid_losses.append(epoch_valid_loss)
            print(f'Epoch {epoch} train MSE: {epoch_train_loss:.7f}, valid MSE: {epoch_valid_loss:.7f}, '
                  f'expert load (block 1): {self.model.expert_load[0].tolist()}')
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
            print(f'WARNING: MAGE predictions are almost constant over time (median std per sensor '
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
        torch.save(self.model.state_dict(), path)   # with the expert balancing bias (a buffer)
        print(f'Model saved to {path}')

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f'Model loaded from {path}')
