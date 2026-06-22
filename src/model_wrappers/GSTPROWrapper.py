import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class GSTPROModel(nn.Module):
    """
    GST-Pro: Graph Spatiotemporal Process for Multivariate Time Series Anomaly Detection
    with Missing Values (Information Fusion 2024, arXiv:2401.05800).

    Architecture (discrete Euler approximation of the Neural CDE formulation):

    1. Learnable graph: A = ReLU(E @ E^T), row-normalised.
       Chebyshev polynomial convolution (order K) mixes node features spatially.

    2. Spatial NCDE (Euler step for each time step t):
         dX_t  = X_proj_t − X_proj_{t−1}          (path derivative / control signal)
         H_t   = H_{t−1} + g_θ(H_{t−1}) ⊙ ChebConv(dX_t, A)

    3. Temporal NCDE (Euler step, H sequence as control):
         dH_t  = H_t − H_{t−1}
         Z_t   = Z_{t−1} + f_θ(Z_{t−1}) ⊙ dH_t

    4. Forecast head: Z_T → ŷ  (next-step prediction for each node)

    5. Masked L1 loss: only observed (non-NaN) positions contribute to the gradient.
       This is the key mechanism for handling missing values without imputation.

    Reference: Koh et al., "Graph spatiotemporal process for multivariate time series
               anomaly detection with missing values", Information Fusion 2024.
    """

    def __init__(self, num_nodes: int, node_features: int,
                 embed_dim: int = 32, hidden_dim: int = 64,
                 K: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.K = K

        # Learnable node embeddings for adaptive graph construction
        self.node_embed = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)

        # Project raw node features to hidden dimension
        self.input_proj = nn.Linear(node_features, hidden_dim)

        # Spatial NCDE function g_θ: H → element-wise gate for the spatial update
        self.spatial_g = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Temporal NCDE function f_θ: Z → element-wise gate for the temporal update
        self.temporal_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Forecast head: final temporal hidden state → next-step node features
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_features),
            nn.Sigmoid(),
        )

    def _build_adj(self) -> torch.Tensor:
        """Compute row-normalised adjacency A = ReLU(E @ E^T) / rowsum."""
        A = torch.relu(self.node_embed @ self.node_embed.T)          # [N, N]
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return A / deg                                                 # [N, N]

    def _cheb_conv(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        K-order Chebyshev graph convolution.
        x: [B, N, hidden]  A: [N, N]  →  [B, N, hidden]
        T_0 = x, T_1 = A@x, T_2 = 2A@T_1 − T_0, …; output = mean of T_0..T_K.
        """
        T_prev = x
        T_curr = torch.einsum('nm,bmd->bnd', A, x)
        out = T_prev + T_curr
        for _ in range(2, self.K + 1):
            T_next = 2.0 * torch.einsum('nm,bmd->bnd', A, T_curr) - T_prev
            out = out + T_next
            T_prev, T_curr = T_curr, T_next
        return out / (self.K + 1)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, N, F] — may contain NaN for missing observations.
        Returns: forecast [B, N, F] for next time step, and nan_mask [B, T, N, F].
        """
        B, T, N, F_in = x.shape

        # Binary mask: 1 where observed, 0 where missing
        obs_mask = (~torch.isnan(x)).float()   # [B, T, N, F]
        x_clean = x.nan_to_num(0.0)            # replace NaN with 0 for computation

        A = self._build_adj()                  # [N, N]

        # Project input sequence: [B, T, N, hidden]
        x_proj = F.relu(self.input_proj(x_clean))

        # Initialise hidden states
        H = torch.zeros(B, N, self.hidden_dim, device=x.device)
        Z = torch.zeros(B, N, self.hidden_dim, device=x.device)

        for t in range(T):
            x_t = x_proj[:, t]                # [B, N, hidden]

            # Control signal: path derivative (finite difference)
            dX = x_t if t == 0 else x_t - x_proj[:, t - 1]   # [B, N, hidden]

            # Spatial NCDE step: H_t = H_{t-1} + g(H_{t-1}) ⊙ ChebConv(dX)
            dX_spatial = self._cheb_conv(dX, A)                # [B, N, hidden]
            H_new = H + self.spatial_g(H) * dX_spatial

            # Temporal NCDE step: Z_t = Z_{t-1} + f(Z_{t-1}) ⊙ (H_t − H_{t-1})
            Z = Z + self.temporal_f(Z) * (H_new - H)

            H = H_new

        forecast = self.forecast_head(Z)       # [B, N, F]
        return forecast, obs_mask


class GSTPROWrapper:

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 embed_dim: int = 32, hidden_dim: int = 64, K: int = 2,
                 dropout: float = 0.1, batch_size: int = 32, device='cpu'):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.slide_win = slide_win
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = device
        self.inference_time = 0
        self._init_model()

    def _init_model(self):
        self.model = GSTPROModel(
            num_nodes=self.num_nodes,
            node_features=self.node_features,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            K=self.K,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        print(f'GSTPROWrapper — device: {self.device}, nodes: {self.num_nodes}, '
              f'features: {self.node_features}, hidden: {self.hidden_dim}, K: {self.K}')

    def _masked_l1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 loss that ignores NaN positions in target."""
        mask = (~torch.isnan(target)).float()
        target_clean = target.nan_to_num(0.0)
        loss = (mask * (pred - target_clean).abs()).sum()
        denom = mask.sum().clamp(min=1.0)
        return loss / denom

    def train(self, train_loader, val_loader, epochs: int):
        train_losses, valid_losses = [], []
        t0 = time.time()

        for epoch in range(epochs):
            self.model.train()
            loss_list = []
            for _, (inputs, labels) in tqdm(enumerate(train_loader),
                                            total=len(train_loader),
                                            desc='Training...'):
                forecast, _ = self.model(inputs)
                loss = self._masked_l1(forecast, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_list.append(loss.item())

            epoch_train_loss = sum(loss_list) / len(loss_list)
            train_losses.append(epoch_train_loss)
            _, _, _, epoch_valid_loss = self.predict(val_loader, mode='valid')
            valid_losses.append(epoch_valid_loss)
            print(f'Epoch {epoch} train loss: {epoch_train_loss:.7f}, '
                  f'valid loss: {epoch_valid_loss:.7f}')

        self.history = {
            'epochs': epochs,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'training_time': time.time() - t0,
        }
        return self.history

    def predict(self, loader, mode: str):
        self.model.eval()
        total_loss, errors, preds = [], [], []
        t0 = time.time()

        with torch.no_grad():
            for inputs, labels in tqdm(loader, total=len(loader), desc='Testing...'):
                forecast, _ = self.model(inputs)
                preds.append(forecast.cpu().numpy())

                # Error: NaN-safe absolute forecast error
                mask = (~torch.isnan(labels)).float()
                labels_clean = labels.nan_to_num(0.0)
                err = mask * (forecast - labels_clean).abs()
                errors.append(err.cpu().numpy())

                total_loss.append(self._masked_l1(forecast, labels).item())

        if mode == 'test':
            self.inference_time = time.time() - t0

        return (
            np.concatenate(preds, axis=0),    # [total, N, F]
            None,                              # no is_nan tracking
            np.concatenate(errors, axis=0),   # [total, N, F]
            sum(total_loss) / len(total_loss),
        )

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f'Model loaded from {path}')