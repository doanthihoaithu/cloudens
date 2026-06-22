import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class USADModel(nn.Module):
    """
    USAD: UnSupervised Anomaly Detection on multivariate time series.

    Two autoencoders share one GRU encoder E. Decoder D1 (AE1) and Decoder
    D2 (AE2) are trained adversarially:
      - AE1 minimises reconstruction AND tries to make AE2 succeed on its
        output (collaborative push toward x).
      - AE2 minimises reconstruction AND tries to make its output of AE1
        diverge from x (adversarial pull).

    This amplifies errors on anomalous inputs: AE1's imperfect output is
    fed back through E → D2, producing a chained score that is high
    whenever the data is unfamiliar to either decoder.

    Training uses a 1/n schedule (n = epoch number) that starts with pure
    reconstruction (n=1 → α=1) and shifts toward adversarial emphasis as
    training progresses.

    Anomaly score: α·|D1(E(x))−y| + β·|D2(E(D1-window))−y|

    Reference: Audibert et al., "USAD: UnSupervised Anomaly Detection on
               Multivariate Time Series", KDD 2020.
    """

    def __init__(self, num_vars: int, slide_win: int,
                 hidden_dim: int = 64, latent_dim: int = 32, n_layers: int = 1):
        super().__init__()
        self.num_vars = num_vars
        self.slide_win = slide_win

        # Shared GRU encoder — encodes the input window → latent z
        self.encoder_gru = nn.GRU(num_vars, hidden_dim, n_layers, batch_first=True)
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)

        # AE1 Decoder: z → next-step prediction
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_vars),
            nn.Sigmoid(),
        )

        # AE2 Decoder: z → next-step prediction (adversarial)
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_vars),
            nn.Sigmoid(),
        )

    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        """x_flat: [B, T, V] → z: [B, latent_dim]"""
        _, h = self.encoder_gru(x_flat)   # [n_layers, B, hidden]
        return self.fc_enc(h[-1])          # [B, latent]

    def forward(self, x: torch.Tensor):
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        x_flat = x.reshape(B, T, self.num_vars)      # [B, T, V]

        # Encode the input window
        z = self.encode(x_flat)                       # [B, latent]

        # AE1: D1(z) → next-step prediction
        pred1 = self.decoder1(z).reshape(B, N, F)     # [B, N, F]

        # AE2: D2(z) → next-step prediction (parallel path)
        pred2 = self.decoder2(z).reshape(B, N, F)     # [B, N, F]

        # Adversarial chain: shift window forward using D1's prediction, re-encode
        # shifted: drop first step, append pred1 as new last step → [B, T, N, F]
        shifted = torch.cat(
            [x[:, 1:, :, :], pred1.detach().unsqueeze(1)], dim=1
        )
        z12 = self.encode(shifted.reshape(B, T, self.num_vars))
        pred12 = self.decoder2(z12).reshape(B, N, F)  # [B, N, F]

        return pred1, pred2, pred12


class USADWrapper:

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 hidden_dim: int = 64, latent_dim: int = 32, n_layers: int = 1,
                 alpha: float = 0.5, batch_size: int = 32, device='cpu'):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.num_vars = num_nodes * node_features
        self.slide_win = slide_win
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.alpha = alpha   # anomaly score weight for pred1 vs pred12
        self.batch_size = batch_size
        self.device = device
        self.inference_time = 0
        self._init_model()

    def _init_model(self):
        self.model = USADModel(
            num_vars=self.num_vars,
            slide_win=self.slide_win,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
        ).to(self.device)

        # Two optimisers: opt1 for E+D1, opt2 for E+D2
        ae1_params = (list(self.model.encoder_gru.parameters()) +
                      list(self.model.fc_enc.parameters()) +
                      list(self.model.decoder1.parameters()))
        ae2_params = (list(self.model.encoder_gru.parameters()) +
                      list(self.model.fc_enc.parameters()) +
                      list(self.model.decoder2.parameters()))
        self.optimizer1 = torch.optim.Adam(ae1_params, lr=1e-3)
        self.optimizer2 = torch.optim.Adam(ae2_params, lr=1e-3)

        print(f'USADWrapper — device: {self.device}, vars: {self.num_vars}, '
              f'hidden: {self.hidden_dim}, latent: {self.latent_dim}')

    def _step_losses(self, pred1, pred2, pred12, labels, epoch_alpha):
        """
        L_AE1 = (1/n)*MSE(pred1, y) + (1-1/n)*MSE(pred12, y)   [E+D1]
        L_AE2 = (1/n)*MSE(pred2, y) - (1-1/n)*MSE(pred12, y)   [E+D2]
        """
        recon1 = F.mse_loss(pred1, labels)
        recon2 = F.mse_loss(pred2, labels)
        adv    = F.mse_loss(pred12, labels)
        beta   = 1.0 - epoch_alpha

        loss_ae1 = epoch_alpha * recon1 + beta * adv
        loss_ae2 = epoch_alpha * recon2 - beta * adv
        return loss_ae1, loss_ae2

    def train(self, train_loader, val_loader, epochs: int):
        train_losses, valid_losses = [], []
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_alpha = 1.0 / epoch    # 1/n schedule: 1 → 1/epochs
            loss_list = []

            for _, (inputs, labels) in tqdm(enumerate(train_loader),
                                            total=len(train_loader),
                                            desc='Training...'):
                # ---- AE1 update (E + D1) ----
                pred1, pred2, pred12 = self.model(inputs)
                loss_ae1, _ = self._step_losses(pred1, pred2, pred12, labels, epoch_alpha)
                self.optimizer1.zero_grad()
                loss_ae1.backward()
                self.optimizer1.step()

                # ---- AE2 update (E + D2) — recompute forward ----
                pred1, pred2, pred12 = self.model(inputs)
                _, loss_ae2 = self._step_losses(pred1, pred2, pred12, labels, epoch_alpha)
                self.optimizer2.zero_grad()
                loss_ae2.backward()
                self.optimizer2.step()

                loss_list.append((loss_ae1 + loss_ae2).item())

            epoch_train_loss = sum(loss_list) / len(loss_list)
            train_losses.append(epoch_train_loss)
            _, _, _, epoch_valid_loss = self.predict(val_loader, mode='valid')
            valid_losses.append(epoch_valid_loss)
            print(f'Epoch {epoch} train loss: {epoch_train_loss:.7f}, '
                  f'valid MSE: {epoch_valid_loss:.7f}')

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
        t0 = time.time()

        with torch.no_grad():
            for inputs, labels in tqdm(loader, total=len(loader), desc='Testing...'):
                pred1, _, pred12 = self.model(inputs)
                # Ensemble anomaly score: α·|AE1 error| + (1-α)·|chain error|
                pred_avg = self.alpha * pred1 + (1.0 - self.alpha) * pred12
                preds.append(pred_avg.cpu().numpy())
                total_loss.append(F.mse_loss(pred_avg, labels).item())
                err = (self.alpha * (pred1 - labels).abs() +
                       (1.0 - self.alpha) * (pred12 - labels).abs())
                errors.append(err.cpu().numpy())

        if mode == 'test':
            self.inference_time = time.time() - t0

        return (
            np.concatenate(preds, axis=0),    # [total, N, F]
            None,                              # no is_nan tracking
            np.concatenate(errors, axis=0),   # [total, N, F]
            sum(total_loss) / len(total_loss),
        )

    def save(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'opt1': self.optimizer1.state_dict(),
            'opt2': self.optimizer2.state_dict(),
        }, path)
        print(f'Model saved to {path}')

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer1.load_state_dict(ckpt['opt1'])
        self.optimizer2.load_state_dict(ckpt['opt2'])
        print(f'Model loaded from {path}')