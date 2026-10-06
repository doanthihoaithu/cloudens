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
      • Prior-association P: a Gaussian kernel centred at each position,
        whose scale σ is projected from the input (one σ per position and
        head).  P is a smooth, local prior — it can only concentrate near
        the query position.
      • Series-association S: standard scaled dot-product attention.
        Normal points can spread attention globally; anomalies tend to
        concentrate locally (similar to P), reducing the discrepancy.

    Association discrepancy = symmetric KL(P ‖ S) per position, averaged
    over heads.  It is large for normal points and small for anomalous
    ones — used as a training signal via minimax and as a score weight.

    Two copies of the discrepancy are returned with different stop-gradients
    (as in the official implementation):
      • series_disc: P detached → gradients only reach the series branch
      • prior_disc:  S detached → gradients only reach the prior branch

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

        # Input-dependent scale of the Gaussian prior — one σ per position and head
        self.W_sigma = nn.Linear(d_model, nhead)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _prior(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian prior P: [B, H, T, T], row-normalised."""
        T = x.size(1)
        # Bound σ to (0, 2] as in the official implementation
        sigma = torch.sigmoid(5.0 * self.W_sigma(x)) + 1e-5       # [B, T, H]
        sigma = (torch.pow(3.0, sigma) - 1.0).transpose(1, 2)     # [B, H, T]
        idx = torch.arange(T, device=x.device).float()
        dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)).pow(2)      # [T, T]
        P = torch.exp(-dist2 / (2.0 * sigma.unsqueeze(-1).pow(2))) \
            / (math.sqrt(2.0 * math.pi) * sigma.unsqueeze(-1))     # [B, H, T, T]
        return P / (P.sum(dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def _sym_kl(P: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Symmetric KL per position, averaged over heads: [B, H, T, T] → [B, T]."""
        eps = 1e-8
        Pc, Sc = P.clamp(min=eps), S.clamp(min=eps)
        kl_ps = (Pc * (Pc.log() - Sc.log())).sum(-1)             # [B, H, T]
        kl_sp = (Sc * (Sc.log() - Pc.log())).sum(-1)             # [B, H, T]
        return (kl_ps + kl_sp).mean(dim=1)                        # [B, T]

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, d_model]
        Returns: x_out [B, T, d_model], series_disc [B, T], prior_disc [B, T]
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

        # Prior association P
        P = self._prior(x)                                        # [B, H, T, T]

        # Association discrepancy with stop-gradient on one side
        series_disc = self._sym_kl(P.detach(), S)                 # [B, T]
        prior_disc = self._sym_kl(P, S.detach())                  # [B, T]

        # Attention-weighted value aggregation
        out = torch.matmul(S, V).transpose(1, 2).reshape(B, T, -1)   # [B, T, d]
        out = self.W_O(out)

        x = self.norm1(x + self.drop(out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, series_disc, prior_disc


# ── Anomaly Transformer Model ─────────────────────────────────────────────────

class AnomalyTransformerModel(nn.Module):
    """
    Full Anomaly Transformer: input projection → positional encoding →
    stack of Anomaly-Attention layers → reconstruction head + forecast head.

    The model returns three values:
      • forecast [B, N, F]: next-step prediction from the last token
      • recon    [B, T, N, F]: full-window reconstruction
      • series_disc [B, T]: association discrepancy averaged over layers,
                            P detached (for the maximise phase / scoring)
      • prior_disc  [B, T]: same values, S detached (for the minimise phase)
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

        series_disc, prior_disc = 0.0, 0.0
        for layer in self.layers:
            h, s_disc, p_disc = layer(h)
            series_disc = series_disc + s_disc
            prior_disc = prior_disc + p_disc
        series_disc = series_disc / len(self.layers)      # [B, T], average over layers
        prior_disc = prior_disc / len(self.layers)

        recon = self.recon_head(h).reshape(B, T, N, F_in)            # [B, T, N, F]
        forecast = self.forecast_head(h[:, -1]).reshape(B, N, F_in)  # [B, N, F]

        return forecast, recon, series_disc, prior_disc


# ── Wrapper ───────────────────────────────────────────────────────────────────

class AnomalyTransformerWrapper:
    """
    Minimax training strategy (from the paper), in one forward pass:

    Maximise phase — series branch (P detached):
        MSE(recon, x) + MSE(forecast, y) − λ · AssocDisc
        → series attention spreads widely for normal windows.

    Minimise phase — prior branch (S detached):
        MSE(recon, x) + MSE(forecast, y) + λ · AssocDisc
        → the Gaussian prior P chases S.

    Both losses are back-propagated and applied in one optimiser step.
    After training, normal points have large AssocDisc (S spreads, P
    can't follow) while anomalous points have small AssocDisc (S
    concentrates locally, P approximates well).

    Anomaly score at inference (score_mode):
      • 'forecast'    : |forecast − labels|, consistent with all other
                        pipeline wrappers (default).
      • 'association' : |forecast − labels| weighted by the paper's
                        association-based criterion Softmax(−AssocDisc · τ).
                        Each window contributes the discrepancy of its last
                        position; the softmax runs over all scored time
                        points and is rescaled so the weights average to 1.
                        (A softmax inside each window, as in the official
                        code, is near one-hot and zeroes the last position.)
    """

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 d_model: int = 64, nhead: int = 4, n_layers: int = 3,
                 lambda_: float = 0.1, dropout: float = 0.1,
                 score_mode: str = 'forecast', temperature: float = 1.0,
                 batch_size: int = 32, device='cpu'):
        assert score_mode in ('forecast', 'association'), f'Unknown score_mode: {score_mode}'
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.num_vars = num_nodes * node_features
        self.slide_win = slide_win
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = n_layers
        self.lambda_ = lambda_
        self.dropout = dropout
        self.score_mode = score_mode
        self.temperature = temperature
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f'AnomalyTransformerWrapper — device: {self.device}, '
              f'vars: {self.num_vars}, d_model: {self.d_model}, '
              f'nhead: {self.nhead}, layers: {self.n_layers}, λ: {self.lambda_}, '
              f'score: {self.score_mode}  [{n_params} params]')

    def _minimax_step(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        One minimax step on a single batch.  Returns the forecast MSE (for logging).
        """
        forecast, recon, series_disc, prior_disc = self.model(inputs)
        rec_loss = self.loss_fn(recon, inputs) + self.loss_fn(forecast, labels)

        loss_max = rec_loss - self.lambda_ * series_disc.mean()   # maximise discrepancy
        loss_min = rec_loss + self.lambda_ * prior_disc.mean()    # minimise discrepancy

        self.optimizer.zero_grad()
        (loss_max + loss_min).backward()
        self.optimizer.step()

        return self.loss_fn(forecast.detach(), labels).item()

    def _association_weight(self, last_disc: np.ndarray) -> np.ndarray:
        """Softmax(−AssocDisc · τ) over all time points, mean-normalised: [total] → [total]."""
        z = -last_disc * self.temperature
        w = np.exp(z - z.max())
        return w / w.mean()

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
        total_loss, errors, preds, last_discs = [], [], [], []
        t0 = time.time()

        with torch.no_grad():
            for inputs, labels in tqdm(loader, total=len(loader), desc='Testing...'):
                forecast, _, series_disc, _ = self.model(inputs)
                preds.append(forecast.cpu().numpy())
                total_loss.append(self.loss_fn(forecast, labels).item())
                errors.append((forecast - labels).abs().cpu().numpy())
                last_discs.append(series_disc[:, -1].cpu().numpy())

        errors = np.concatenate(errors, axis=0)                   # [total, N, F]
        if self.score_mode == 'association':
            weight = self._association_weight(np.concatenate(last_discs, axis=0))
            errors = errors * weight[:, None, None]

        if mode == 'test':
            self.inference_time = time.time() - t0

        return (
            np.concatenate(preds,   axis=0),   # [total, N, F]
            None,                               # no is_nan tracking
            errors,                             # [total, N, F]
            sum(total_loss) / len(total_loss),
        )

    def save(self, path: str):
        torch.save({
            'model':     self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f'Model saved to {path}')

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if 'optimizer' not in ckpt:
            raise RuntimeError(f'{path} was saved by the old AnomalyTransformerWrapper '
                               f'(global σ per head) and is incompatible — retrain the model.')
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f'Model loaded from {path}')
