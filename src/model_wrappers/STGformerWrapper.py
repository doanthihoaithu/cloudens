import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _mlp(dim: int, mlp_ratio: float, dropout: float) -> nn.Sequential:
    """fc → ReLU → dropout → fc → dropout (timm's Mlp, as used by the official code)."""
    hidden = int(dim * mlp_ratio)
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, dim), nn.Dropout(dropout),
    )


def _linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Linear attention with L2-normalised queries/keys and a self-loop term.
    q, k, v: [B', L, H, D] → [B', L, H, D].  Cost is linear in the sequence length L.
    """
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    L = q.shape[1]
    kv = torch.einsum('blhm,blhd->bhmd', k, v)
    num = torch.einsum('blhm,bhmd->blhd', q, kv) + L * v
    denom = torch.einsum('blhm,bhm->blh', q, k.sum(dim=1)).unsqueeze(-1) + L
    return num / denom


class STGAttention(nn.Module):
    """
    Spatiotemporal linear attention with shared Q/K/V:
    attention over the N nodes at each time step, and over the T steps of
    each node; both outputs are concatenated and projected back to C.
    """

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        assert model_dim % num_heads == 0, f'model_dim ({model_dim}) must be divisible by num_heads ({num_heads})'
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out_proj = nn.Linear(2 * model_dim, model_dim)

    def _attend(self, q, k, v):
        # q, k, v: [B, S, L, C] — attention runs along L, independently for each of the S rows
        B, S, L, C = q.shape
        split = lambda t: t.reshape(B * S, L, self.num_heads, self.head_dim)
        return _linear_attention(split(q), split(k), split(v)).reshape(B, S, L, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        out_s = self._attend(q, k, v)                                        # over nodes
        out_t = self._attend(q.transpose(1, 2), k.transpose(1, 2),
                             v.transpose(1, 2)).transpose(1, 2)              # over time
        return self.out_proj(torch.cat([out_s, out_t], dim=-1))             # [B, T, N, C]


class STGBlock(nn.Module):
    """
    STG attention block: graph propagation to orders 0..K-1, attention on each
    order, combined by recursive gating  x_glo += a_k(z_k) ⊙ g_k(a_{k-1}) · s_k.
    The gates g_k start at zero, so the block begins as an identity mapping.
    """

    SCALES = [1.0, 0.01, 0.001]

    def __init__(self, model_dim: int, num_heads: int, order: int = 2,
                 mlp_ratio: float = 2, dropout: float = 0.1, prop_dropout: float = 0.2):
        super().__init__()
        assert 1 <= order <= len(self.SCALES), f'order must be in [1, {len(self.SCALES)}]'
        self.order = order
        self.prop_dropout = nn.Dropout(prop_dropout)
        self.attn = nn.ModuleList([STGAttention(model_dim, num_heads) for _ in range(order)])
        self.gates = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(order)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.ffn = _mlp(model_dim, mlp_ratio, dropout)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C], graph: [T, N, N]
        z, c, x_glo = x, x, x
        for k in range(self.order):
            if k > 0:
                z = self.prop_dropout(torch.einsum('tnm,btmc->btnc', graph, z))   # A^k x
            att = self.attn[k](z)
            x_glo = x_glo + att * self.gates[k](c) * self.SCALES[k]
            c = att
        x = self.ln1(x + self.dropout(x_glo))
        return self.ln2(x + self.dropout(self.ffn(x)))


class STGformerModel(nn.Module):
    """
    STGformer: Efficient Spatiotemporal Graph Transformer, adapted to
    next-step forecasting.

    Tokens = input projection ‖ adaptive spatiotemporal embedding.  A graph
    per time step is learned from the adaptive embedding,
    A_t = Softmax(ReLU(E_t E_tᵀ)), and drives the propagation inside one STG
    attention block.  The T tokens of each node are then flattened, projected
    and refined by residual MLPs before the forecasting head.

    Follows the official implementation (github.com/Dreamzz5/STGformer),
    without the time-of-day / day-of-week embeddings (not available here).

    Reference: Wang et al., "STGformer: Efficient Spatiotemporal Graph
               Transformer for Traffic Forecasting", arXiv:2410.00385, 2024.
    """

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 input_embedding_dim: int = 24, adaptive_embedding_dim: int = 40,
                 num_heads: int = 4, num_layers: int = 3, order: int = 2,
                 mlp_ratio: float = 2, dropout: float = 0.1, adaptive_dropout: float = 0.3,
                 out_features: int = None):
        super().__init__()
        out_features = out_features or node_features
        model_dim = input_embedding_dim + adaptive_embedding_dim

        self.input_proj = nn.Linear(node_features, input_embedding_dim)
        self.adaptive_embedding = nn.Parameter(torch.empty(slide_win, num_nodes, adaptive_embedding_dim))
        nn.init.xavier_uniform_(self.adaptive_embedding)
        self.adaptive_dropout = nn.Dropout(adaptive_dropout)
        self.temporal_proj = nn.Linear(model_dim, model_dim)

        self.stg_block = STGBlock(model_dim, num_heads, order, mlp_ratio, dropout)

        self.encoder_proj = nn.Linear(slide_win * model_dim, model_dim)
        self.encoder = nn.ModuleList([_mlp(model_dim, mlp_ratio, dropout) for _ in range(num_layers)])
        self.output_proj = nn.Linear(model_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F]
        B = x.size(0)
        adp = self.adaptive_dropout(self.adaptive_embedding).expand(B, -1, -1, -1)
        h = torch.cat([self.input_proj(x), adp], dim=-1)                    # [B, T, N, C]
        h = self.temporal_proj(h)

        E = self.adaptive_embedding
        graph = torch.softmax(F.relu(E @ E.transpose(1, 2)), dim=-1)        # [T, N, N]
        h = self.stg_block(h, graph)

        h = self.encoder_proj(h.transpose(1, 2).flatten(-2))                # [B, N, C]
        for layer in self.encoder:
            h = h + layer(h)
        return torch.sigmoid(self.output_proj(h))                           # [B, N, F_out]


class STGformerWrapper:

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 input_embedding_dim: int = 24, adaptive_embedding_dim: int = 40,
                 num_heads: int = 4, num_layers: int = 3, order: int = 2,
                 dropout: float = 0.1,
                 null_padding_feature: bool = False, null_padding_target: bool = False,
                 batch_size: int = 32, device='cpu'):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.slide_win = slide_win
        self.input_embedding_dim = input_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.order = order
        self.dropout = dropout
        # Null padding appends an is_nan channel to the inputs and/or targets
        self.null_padding_feature = null_padding_feature
        self.null_padding_target = null_padding_target
        self.batch_size = batch_size
        self.device = device
        self.inference_time = 0
        self._init_model()

    def _init_model(self):
        self.model = STGformerModel(
            num_nodes=self.num_nodes,
            node_features=self.node_features + int(self.null_padding_feature),
            slide_win=self.slide_win,
            input_embedding_dim=self.input_embedding_dim,
            adaptive_embedding_dim=self.adaptive_embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            order=self.order,
            dropout=self.dropout,
            out_features=self.node_features + int(self.null_padding_target),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=3e-4)
        self.loss_fn = nn.MSELoss()
        print(f'STGformerWrapper — device: {self.device}, nodes: {self.num_nodes}, '
              f'features: {self.node_features}, '
              f'model_dim: {self.input_embedding_dim + self.adaptive_embedding_dim}, '
              f'heads: {self.num_heads}, order: {self.order}, '
              f'null padding feature/target: {self.null_padding_feature}/{self.null_padding_target}')

    def train(self, train_loader, val_loader, epochs: int):
        train_losses, valid_losses = [], []
        t0 = time.time()

        for epoch in range(epochs):
            self.model.train()
            loss_list = []
            for _, (inputs, labels) in tqdm(enumerate(train_loader),
                                            total=len(train_loader),
                                            desc='Training...'):
                y_hat = self.model(inputs)
                loss = self.loss_fn(y_hat, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_list.append(loss.item())

            epoch_train_loss = sum(loss_list) / len(loss_list)
            train_losses.append(epoch_train_loss)
            _, _, _, epoch_valid_loss = self.predict(val_loader, mode='valid')
            valid_losses.append(epoch_valid_loss)
            print(f'Epoch {epoch} train RMSE: {epoch_train_loss:.7f}, valid RMSE: {epoch_valid_loss:.7f}')

        training_time = time.time() - t0
        self.history = {
            'epochs': epochs,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'training_time': training_time,
        }
        return self.history

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
