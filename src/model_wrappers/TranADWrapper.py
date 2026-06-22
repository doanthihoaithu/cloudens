import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return self.dropout(x + self.pe[:, :x.size(1)])


class TranADModel(nn.Module):
    """
    TranAD: two-decoder Transformer for next-step forecasting.

    Decoder 1 reconstructs the next step normally. Decoder 2 receives the
    residual error from Decoder 1 as additional context and uses a focus-
    weighted loss to concentrate on what Decoder 1 found hard to predict.

    Reference: Tuli et al., "TranAD: Deep Transformer Networks for Anomaly
               Detection in Multivariate Time Series Data", VLDB 2022.
    """

    def __init__(self, num_vars: int, slide_win: int,
                 d_model: int = 64, nhead: int = 4,
                 n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_vars = num_vars

        self.input_proj = nn.Linear(num_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=slide_win + 10)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model,
                                               dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)

        dec_layer1 = nn.TransformerDecoderLayer(d_model, nhead, 4 * d_model,
                                                dropout, batch_first=True)
        self.decoder1 = nn.TransformerDecoder(dec_layer1, n_layers)

        dec_layer2 = nn.TransformerDecoderLayer(d_model, nhead, 4 * d_model,
                                                dropout, batch_first=True)
        self.decoder2 = nn.TransformerDecoder(dec_layer2, n_layers)

        self.out1 = nn.Linear(d_model, num_vars)
        self.out2 = nn.Linear(d_model, num_vars)

    def forward(self, x: torch.Tensor):
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        x_flat = x.reshape(B, T, self.num_vars)            # [B, T, V]

        # Encode the input window  [B, T, d]
        src = self.pos_enc(self.input_proj(x_flat))        # [B, T, d]
        memory = self.encoder(src)                          # [B, T, d]

        # Decoder 1 — next-step prediction using last encoder state as query
        tgt = memory[:, -1:, :]                            # [B, 1, d]
        out1 = self.decoder1(tgt, memory)                  # [B, 1, d]
        pred1 = torch.sigmoid(self.out1(out1.squeeze(1))).reshape(B, N, F)  # [B, N, F]

        # Decoder 2 — conditioned on residual between input and Decoder 1 output
        # detach() stops Decoder2's gradient from flowing back through pred1 into Decoder1
        residual = x_flat - pred1.detach().reshape(B, 1, -1)  # [B, T, V]
        res_ctx = self.pos_enc(self.input_proj(residual))  # [B, T, d]
        tgt2 = tgt + res_ctx[:, -1:, :]                   # focus query incorporates residual
        out2 = self.decoder2(tgt2, memory + res_ctx)       # [B, 1, d]
        pred2 = torch.sigmoid(self.out2(out2.squeeze(1))).reshape(B, N, F)  # [B, N, F]

        return pred1, pred2


class TranADWrapper:

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 d_model: int = 64, nhead: int = 4, n_layers: int = 1,
                 dropout: float = 0.1, batch_size: int = 32, device='cpu'):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.num_vars = num_nodes * node_features
        self.slide_win = slide_win
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = device
        self.inference_time = 0
        self._init_model()

    def _init_model(self):
        self.model = TranADModel(
            num_vars=self.num_vars,
            slide_win=self.slide_win,
            d_model=self.d_model,
            nhead=self.nhead,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        print(f'TranADWrapper — device: {self.device}, vars: {self.num_vars}, '
              f'd_model: {self.d_model}, nhead: {self.nhead}')

    def _compute_loss(self, pred1: torch.Tensor, pred2: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
        """Phase-1 MSE (Decoder1) + phase-2 focus-weighted MSE (Decoder2)."""
        loss1 = self.loss_fn(pred1, labels)

        # Focus weight: higher where Decoder1 made a larger error
        # 1/(1+e) → weight is inversely proportional to D1 error magnitude
        focus = 1.0 / (1.0 + (pred1.detach() - labels).abs())
        loss2 = (focus * (pred2 - labels).pow(2)).mean()

        return loss1 + loss2

    def train(self, train_loader, val_loader, epochs: int):
        train_losses, valid_losses = [], []
        t0 = time.time()

        for epoch in range(epochs):
            self.model.train()
            loss_list = []
            for _, (inputs, labels) in tqdm(enumerate(train_loader),
                                            total=len(train_loader),
                                            desc='Training...'):
                pred1, pred2 = self.model(inputs)
                loss = self._compute_loss(pred1, pred2, labels)
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
        t0 = time.time()

        with torch.no_grad():
            for inputs, labels in tqdm(loader, total=len(loader), desc='Testing...'):
                pred1, pred2 = self.model(inputs)
                pred_avg = (pred1 + pred2) / 2                # ensemble both decoders
                preds.append(pred_avg.cpu().numpy())
                total_loss.append(self.loss_fn(pred_avg, labels).item())
                # Anomaly score: average absolute error across both decoders
                err = ((pred1 - labels).abs() + (pred2 - labels).abs()) / 2
                errors.append(err.cpu().numpy())

        if mode == 'test':
            self.inference_time = time.time() - t0

        return (
            np.concatenate(preds, axis=0),          # [total, N, F]
            None,                                   # no is_nan tracking
            np.concatenate(errors, axis=0),         # [total, N, F]
            sum(total_loss) / len(total_loss),
        )

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f'Model loaded from {path}')