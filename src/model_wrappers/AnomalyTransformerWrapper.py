import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Anomaly-Attention Layer ───────────────────────────────────────────────────

class AnomalyAttentionLayer(nn.Module):
    """
    One Anomaly-Attention block from the Anomaly Transformer paper.

    Computes two attention distributions in parallel:
      • Prior-association P: a learnable Gaussian kernel centred at each
        position, parameterised by a per-head scale σ.  P is a smooth,
        local prior — it can only concentrate near the query position.
      • Series-association S: standard scaled dot-product attention.
        Normal points can spread attention globally; anomalies tend to
        concentrate locally (similar to P), reducing the discrepancy.

    Association discrepancy = symmetric KL(P ‖ S) averaged over batch,
    heads, and positions.  It is large for normal windows and small for
    anomalous ones — used as an auxiliary training signal via minimax.

    Reference: Xu et al., "Anomaly Transformer: Time Series Anomaly
               Detection with Association Discrepancy", ICLR 2022.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f'd_model ({d_model}) must be divisible by nhead ({nhead})'
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Learnable log-scale for the Gaussian prior — one σ per attention head.
        # Initialised near 1.0 so the prior starts as a moderate local kernel.
        self.log_sigma = nn.Parameter(torch.zeros(nhead))

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _prior(self, T: int, device) -> torch.Tensor:
        """Gaussian prior P: [H, T, T], row-normalised."""
        sigma = self.log_sigma.exp().clamp(min=0.01)              # [H]
        idx = torch.arange(T, device=device).float()
        dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)).pow(2)      # [T, T]
        P = torch.exp(
            -dist2.unsqueeze(0) / (2.0 * sigma.pow(2).view(-1, 1, 1) + 1e-8)
        )                                                          # [H, T, T]
        return P / (P.sum(dim=-1, keepdim=True) + 1e-8)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, d_model]
        Returns: x_out [B, T, d_model], assoc_disc (scalar tensor)
        """
        B, T, _ = x.shape

        Q = self.W_Q(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B,H,T,d_h]
        K = self.W_K(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # Series association S: standard softmax attention
        S = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim),
            dim=-1,
        )                                                          # [B, H, T, T]

        # Prior association P broadcast to batch
        P = self._prior(T, x.device)                              # [H, T, T]
        P_exp = P.unsqueeze(0).expand(B, -1, -1, -1)             # [B, H, T, T]

        # Symmetric KL divergence: KL(P‖S) + KL(S‖P), averaged to scalar
        eps = 1e-8
        Pc = P_exp.clamp(min=eps)
        Sc = S.clamp(min=eps)
        kl_ps = (Pc * (Pc.log() - Sc.log())).sum(-1)             # [B, H, T]
        kl_sp = (Sc * (Sc.log() - Pc.log())).sum(-1)             # [B, H, T]
        assoc_disc = (kl_ps + kl_sp).mean()                       # scalar

        # Attention-weighted value aggregation
        out = torch.matmul(S, V).transpose(1, 2).reshape(B, T, -1)   # [B, T, d]
        out = self.W_O(out)

        x = self.norm1(x + self.drop(out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, assoc_disc


# ── Anomaly Transformer Model ─────────────────────────────────────────────────

class AnomalyTransformerModel(nn.Module):
    """
    Full Anomaly Transformer: input projection → positional encoding →
    stack of Anomaly-Attention layers → reconstruction head + forecast head.

    The model returns three values:
      • forecast [B, N, F]: next-step prediction from the last token
      • recon    [B, T, N, F]: full-window reconstruction
      • assoc_disc: mean association discrepancy across all layers (scalar)
    """

    def __init__(self, num_vars: int, slide_win: int,
                 d_model: int = 64, nhead: int = 4,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_vars = num_vars
        self.slide_win = slide_win

        self.input_proj = nn.Linear(num_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=slide_win + 16)

        self.layers = nn.ModuleList([
            AnomalyAttentionLayer(d_model, nhead, dropout)
            for _ in range(n_layers)
        ])

        # Reconstruction: all T positions → V features each
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, num_vars),
            nn.Sigmoid(),
        )

        # Forecast: last token → next-step V features
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, num_vars),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, N, F]
        B, T, N, F_in = x.shape
        x_flat = x.reshape(B, T, self.num_vars)           # [B, T, V]

        h = self.pos_enc(self.input_proj(x_flat))         # [B, T, d]

        total_disc = 0.0
        for layer in self.layers:
            h, disc = layer(h)
            total_disc = total_disc + disc
        total_disc = total_disc / len(self.layers)        # average over layers

        recon = self.recon_head(h).reshape(B, T, N, F_in)            # [B, T, N, F]
        forecast = self.forecast_head(h[:, -1]).reshape(B, N, F_in)  # [B, N, F]

        return forecast, recon, total_disc


# ── Wrapper ───────────────────────────────────────────────────────────────────

class AnomalyTransformerWrapper:
    """
    Minimax training strategy (from the paper):

    Phase 1 — model params (excluding σ):
        minimise  MSE(recon, x) + MSE(forecast, y) − λ · AssocDisc
        → maximises discrepancy so series attention spreads widely for
          normal windows.

    Phase 2 — σ (Gaussian prior scale) only:
        minimise  λ · AssocDisc
        → shrinks the discrepancy by making P chase S.

    After training, normal windows have large AssocDisc (S spreads, P
    can't follow) while anomalous windows have small AssocDisc (S
    concentrates locally, P approximates well).

    Anomaly score at inference: |forecast − labels| (per-timestep
    forecast error, consistent with all other pipeline wrappers).
    """

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 d_model: int = 64, nhead: int = 4, n_layers: int = 3,
                 lambda_: float = 0.1, dropout: float = 0.1,
                 batch_size: int = 32, device='cpu'):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.num_vars = num_nodes * node_features
        self.slide_win = slide_win
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = n_layers
        self.lambda_ = lambda_
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = device
        self.inference_time = 0
        self._init_model()

    def _init_model(self):
        self.model = AnomalyTransformerModel(
            num_vars=self.num_vars,
            slide_win=self.slide_win,
            d_model=self.d_model,
            nhead=self.nhead,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Separate parameter groups for minimax optimisation
        sigma_params = [p for n, p in self.model.named_parameters() if 'log_sigma' in n]
        main_params  = [p for n, p in self.model.named_parameters() if 'log_sigma' not in n]

        self.optimizer_main  = torch.optim.Adam(main_params,  lr=1e-3)
        self.optimizer_sigma = torch.optim.Adam(sigma_params, lr=1e-3)
        self.loss_fn = nn.MSELoss()

        n_sigma = sum(p.numel() for p in sigma_params)
        n_main  = sum(p.numel() for p in main_params)
        print(f'AnomalyTransformerWrapper — device: {self.device}, '
              f'vars: {self.num_vars}, d_model: {self.d_model}, '
              f'nhead: {self.nhead}, layers: {self.n_layers}, λ: {self.lambda_}  '
              f'[main: {n_main} params, σ: {n_sigma} params]')

    def _minimax_step(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        One minimax step on a single batch.  Returns the forecast MSE (for logging).
        """
        # ── Phase 1: main params — maximise discrepancy ─────────────────────
        self.model.zero_grad()
        forecast, recon, assoc_disc = self.model(inputs)
        loss_main = (self.loss_fn(recon, inputs)
                     + self.loss_fn(forecast, labels)
                     - self.lambda_ * assoc_disc)
        loss_main.backward()
        self.optimizer_main.step()

        # ── Phase 2: σ params — minimise discrepancy ────────────────────────
        self.model.zero_grad()
        _, _, assoc_disc2 = self.model(inputs)
        loss_sigma = self.lambda_ * assoc_disc2
        loss_sigma.backward()
        self.optimizer_sigma.step()

        self.model.zero_grad()
        return self.loss_fn(forecast.detach(), labels).item()

    def train(self, train_loader, val_loader, epochs: int):
        train_losses, valid_losses = [], []
        t0 = time.time()

        for epoch in range(epochs):
            self.model.train()
            loss_list = []
            for _, (inputs, labels) in tqdm(enumerate(train_loader),
                                            total=len(train_loader),
                                            desc='Training...'):
                loss_list.append(self._minimax_step(inputs, labels))

            epoch_train_loss = sum(loss_list) / len(loss_list)
            train_losses.append(epoch_train_loss)
            _, _, _, epoch_valid_loss = self.predict(val_loader, mode='valid')
            valid_losses.append(epoch_valid_loss)
            print(f'Epoch {epoch} train forecast MSE: {epoch_train_loss:.7f}, '
                  f'valid MSE: {epoch_valid_loss:.7f}')

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
                forecast, _, _ = self.model(inputs)
                preds.append(forecast.cpu().numpy())
                total_loss.append(self.loss_fn(forecast, labels).item())
                errors.append((forecast - labels).abs().cpu().numpy())

        if mode == 'test':
            self.inference_time = time.time() - t0

        return (
            np.concatenate(preds,   axis=0),   # [total, N, F]
            None,                               # no is_nan tracking
            np.concatenate(errors,  axis=0),   # [total, N, F]
            sum(total_loss) / len(total_loss),
        )

    def save(self, path: str):
        torch.save({
            'model':       self.model.state_dict(),
            'opt_main':    self.optimizer_main.state_dict(),
            'opt_sigma':   self.optimizer_sigma.state_dict(),
        }, path)
        print(f'Model saved to {path}')

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer_main.load_state_dict(ckpt['opt_main'])
        self.optimizer_sigma.load_state_dict(ckpt['opt_sigma'])
        print(f'Model loaded from {path}')