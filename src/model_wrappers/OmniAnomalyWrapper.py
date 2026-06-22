import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class OmniAnomalyModel(nn.Module):
    """
    OmniAnomaly: Stochastic Recurrent Neural Network for multivariate time-series
    anomaly detection. A GRU encodes the input window into a Gaussian latent space
    (VAE-style); a second GRU decodes the sampled latent vector into the next-step
    prediction. At inference the posterior mean is used, giving a deterministic score.

    Training objective: ELBO = reconstruction MSE + β * KL(q(z|x) ‖ p(z))

    Reference: Su et al., "Robust Anomaly Detection for Multivariate Time Series
               through Stochastic Recurrent Neural Network", KDD 2019.
    """

    def __init__(self, num_vars: int, hidden_dim: int = 64,
                 latent_dim: int = 16, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        enc_dropout = dropout if n_layers > 1 else 0.0
        dec_dropout = dropout if n_layers > 1 else 0.0

        # Encoder GRU: maps input window → hidden state h
        self.encoder_gru = nn.GRU(num_vars, hidden_dim, n_layers,
                                   batch_first=True, dropout=enc_dropout)

        # Posterior: h → (μ, log σ²) of latent z
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder GRU: z → hidden state g
        self.decoder_gru = nn.GRU(latent_dim, hidden_dim, n_layers,
                                   batch_first=True, dropout=dec_dropout)

        # Output head: g → next-step prediction x̂
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_vars),
            nn.Sigmoid(),
        )

    def encode(self, x_flat: torch.Tensor):
        """x_flat: [B, T, V] → μ, log σ² each [B, latent_dim]"""
        _, h = self.encoder_gru(x_flat)   # h: [n_layers, B, hidden]
        h_top = h[-1]                      # last layer: [B, hidden]
        return self.fc_mu(h_top), self.fc_logvar(h_top)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z during training; use μ deterministically at inference."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim] → x̂: [B, V]"""
        z_seq = z.unsqueeze(1)             # [B, 1, latent]
        out, _ = self.decoder_gru(z_seq)   # [B, 1, hidden]
        return self.out(out.squeeze(1))    # [B, V]

    def forward(self, x: torch.Tensor):
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        x_flat = x.reshape(B, T, N * F)   # [B, T, V]

        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        pred = self.decode(z).reshape(B, N, F)  # [B, N, F]
        return pred, mu, logvar


class OmniAnomalyWrapper:

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 hidden_dim: int = 64, latent_dim: int = 16,
                 n_layers: int = 2, dropout: float = 0.1,
                 beta: float = 0.1, batch_size: int = 32, device='cpu'):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.num_vars = num_nodes * node_features
        self.slide_win = slide_win
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.beta = beta           # KL weight — controls regularisation strength
        self.batch_size = batch_size
        self.device = device
        self.inference_time = 0
        self._init_model()

    def _init_model(self):
        self.model = OmniAnomalyModel(
            num_vars=self.num_vars,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.recon_loss_fn = nn.MSELoss()
        print(f'OmniAnomalyWrapper — device: {self.device}, vars: {self.num_vars}, '
              f'hidden: {self.hidden_dim}, latent: {self.latent_dim}, β: {self.beta}')

    def _elbo_loss(self, pred: torch.Tensor, labels: torch.Tensor,
                   mu: torch.Tensor, logvar: torch.Tensor):
        """ELBO = reconstruction MSE + β * KL(N(μ,σ²) ‖ N(0,1))"""
        recon = self.recon_loss_fn(pred, labels)
        # Closed-form KL for diagonal Gaussian vs standard normal
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * kl

    def train(self, train_loader, val_loader, epochs: int):
        train_losses, valid_losses = [], []
        t0 = time.time()

        for epoch in range(epochs):
            self.model.train()
            loss_list = []
            for _, (inputs, labels) in tqdm(enumerate(train_loader),
                                            total=len(train_loader),
                                            desc='Training...'):
                pred, mu, logvar = self.model(inputs)
                loss = self._elbo_loss(pred, labels, mu, logvar)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_list.append(loss.item())

            epoch_train_loss = sum(loss_list) / len(loss_list)
            train_losses.append(epoch_train_loss)
            _, _, _, epoch_valid_loss = self.predict(val_loader, mode='valid')
            valid_losses.append(epoch_valid_loss)
            print(f'Epoch {epoch} train ELBO: {epoch_train_loss:.7f}, '
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
                # reparameterize returns μ in eval mode → deterministic score
                pred, _, _ = self.model(inputs)
                preds.append(pred.cpu().numpy())
                total_loss.append(self.recon_loss_fn(pred, labels).item())
                errors.append((pred - labels).abs().cpu().numpy())

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