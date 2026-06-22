import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class MTADGATModel(nn.Module):
    """
    MTAD-GAT: Multivariate Time-series Anomaly Detection via Graph Attention Networks.

    Two parallel self-attention modules replace the original GAT layers:
      - Feature-GAT: at each time step, each node attends to all other nodes
        (captures inter-variable correlations).
      - Time-GAT: for each node, each time step attends to all other time steps
        (captures intra-variable temporal dependencies).

    Their outputs are concatenated and fed into a shared per-node GRU, which
    drives two prediction heads trained jointly:
      - Forecasting head: predicts the next time step (next-step loss).
      - Reconstruction head: reconstructs the input window (self-supervised).

    Anomaly score at inference: absolute error of the forecasting head vs. label.

    Reference: Zhao et al., "Multivariate Time-series Anomaly Detection via
               Graph Attention Networks", ICDM 2020.
    """

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 d_model: int = 32, nhead: int = 2,
                 gru_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features

        # Project raw features into attention dimension
        self.input_proj = nn.Linear(node_features, d_model)

        # Feature-GAT: self-attention across N nodes at each time step
        self.feature_gat = nn.MultiheadAttention(d_model, nhead,
                                                  dropout=dropout, batch_first=True)
        self.feature_norm = nn.LayerNorm(d_model)

        # Time-GAT: self-attention across T time steps for each node
        self.time_gat = nn.MultiheadAttention(d_model, nhead,
                                               dropout=dropout, batch_first=True)
        self.time_norm = nn.LayerNorm(d_model)

        # GRU: processes each node's time series (input = concatenated GAT outputs)
        self.gru = nn.GRU(2 * d_model, gru_hidden, batch_first=True)

        # Forecasting head → next-step prediction
        self.forecast_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(),
            nn.Linear(gru_hidden, node_features),
            nn.Sigmoid(),
        )

        # Reconstruction head → window reconstruction (per time step)
        self.recon_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(),
            nn.Linear(gru_hidden, node_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, N, F]
        B, T, N, F_in = x.shape

        # Project to d_model
        h = F.relu(self.input_proj(x))        # [B, T, N, d_model]

        # --- Feature-GAT: attend across N for each (B, t) ---
        h_feat = h.reshape(B * T, N, -1)      # [B*T, N, d_model]
        feat_out, _ = self.feature_gat(h_feat, h_feat, h_feat)
        feat_out = self.feature_norm(feat_out + h_feat)     # residual + norm
        feat_out = feat_out.reshape(B, T, N, -1)            # [B, T, N, d_model]

        # --- Time-GAT: attend across T for each (B, n) ---
        h_time = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # [B*N, T, d_model]
        time_out, _ = self.time_gat(h_time, h_time, h_time)
        time_out = self.time_norm(time_out + h_time)        # residual + norm
        time_out = time_out.reshape(B, N, T, -1).permute(0, 2, 1, 3)  # [B, T, N, d_model]

        # --- Concatenate and run GRU per node ---
        combined = torch.cat([feat_out, time_out], dim=-1)  # [B, T, N, 2*d_model]
        gru_in = combined.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # [B*N, T, 2*d]
        gru_out, _ = self.gru(gru_in)                       # [B*N, T, gru_h]

        # --- Forecasting: next-step from last GRU hidden state ---
        last_h = gru_out[:, -1, :].reshape(B, N, -1)       # [B, N, gru_h]
        forecast = self.forecast_head(last_h)               # [B, N, F]

        # --- Reconstruction: all GRU steps → window reconstruction ---
        recon = self.recon_head(gru_out)                    # [B*N, T, F]
        recon = recon.reshape(B, N, T, F_in).permute(0, 2, 1, 3)  # [B, T, N, F]

        return forecast, recon


class MTADGATWrapper:

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 d_model: int = 32, nhead: int = 2, gru_hidden: int = 64,
                 dropout: float = 0.1, batch_size: int = 32, device='cpu'):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.slide_win = slide_win
        self.d_model = d_model
        self.nhead = nhead
        self.gru_hidden = gru_hidden
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = device
        self.inference_time = 0
        self._init_model()

    def _init_model(self):
        self.model = MTADGATModel(
            num_nodes=self.num_nodes,
            node_features=self.node_features,
            slide_win=self.slide_win,
            d_model=self.d_model,
            nhead=self.nhead,
            gru_hidden=self.gru_hidden,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        print(f'MTADGATWrapper — device: {self.device}, nodes: {self.num_nodes}, '
              f'features: {self.node_features}, d_model: {self.d_model}, '
              f'gru_hidden: {self.gru_hidden}')

    def train(self, train_loader, val_loader, epochs: int):
        train_losses, valid_losses = [], []
        t0 = time.time()

        for epoch in range(epochs):
            self.model.train()
            loss_list = []
            for _, (inputs, labels) in tqdm(enumerate(train_loader),
                                            total=len(train_loader),
                                            desc='Training...'):
                forecast, recon = self.model(inputs)

                # Joint loss: next-step forecasting + window reconstruction
                loss_f = self.loss_fn(forecast, labels)
                loss_r = self.loss_fn(recon, inputs)
                loss = loss_f + loss_r

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_list.append(loss.item())

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
                forecast, _ = self.model(inputs)
                preds.append(forecast.cpu().numpy())
                total_loss.append(self.loss_fn(forecast, labels).item())
                errors.append((forecast - labels).abs().cpu().numpy())

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