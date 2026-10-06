import math
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class STBlock(nn.Module):
    """
    One spatio-temporal Transformer block (pre-norm):
      1. temporal self-attention — each node attends over its own T steps
      2. spatial self-attention  — each time step attends over the N nodes
      3. position-wise feed-forward
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.t_norm = nn.LayerNorm(d_model)
        self.t_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.s_norm = nn.LayerNorm(d_model)
        self.s_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, N, d]
        B, T, N, d = h.shape

        # 1. Temporal attention: sequences of length T, one per (batch, node)
        ht = h.permute(0, 2, 1, 3).reshape(B * N, T, d)      # [B*N, T, d]
        x = self.t_norm(ht)
        ht = ht + self.dropout(self.t_attn(x, x, x, need_weights=False)[0])
        h = ht.reshape(B, N, T, d).permute(0, 2, 1, 3)       # [B, T, N, d]

        # 2. Spatial attention: sequences of length N, one per (batch, step)
        hs = h.reshape(B * T, N, d)                          # [B*T, N, d]
        x = self.s_norm(hs)
        hs = hs + self.dropout(self.s_attn(x, x, x, need_weights=False)[0])
        h = hs.reshape(B, T, N, d)

        # 3. Feed-forward
        return h + self.dropout(self.ff(self.ff_norm(h)))


class STformerModel(nn.Module):
    """
    STformer: spatio-temporal Transformer for next-step forecasting.

    Each (time step, node) pair is a token. Tokens get a sinusoidal temporal
    encoding plus a learnable node embedding, then pass through stacked
    blocks of factorised attention (temporal over T, spatial over N). The last
    time step's token of each node predicts that node's next value.
    """

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 d_model: int = 64, nhead: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_features, d_model)
        self.node_emb = nn.Parameter(torch.randn(num_nodes, d_model) * 0.02)

        pe = torch.zeros(slide_win, d_model)
        position = torch.arange(slide_win).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('time_pe', pe.unsqueeze(1))      # [T, 1, d]

        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([STBlock(d_model, nhead, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_layer = nn.Linear(d_model, node_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F]
        T = x.size(1)
        h = self.input_proj(x) + self.time_pe[:T] + self.node_emb   # [B, T, N, d]
        h = self.input_dropout(h)
        for block in self.blocks:
            h = block(h)
        last = self.out_norm(h[:, -1])                        # [B, N, d]
        return torch.sigmoid(self.out_layer(last))            # [B, N, F]


class STformerWrapper:

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 d_model: int = 64, nhead: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, batch_size: int = 32, device='cpu'):
        self.num_nodes = num_nodes
        self.node_features = node_features
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
        self.model = STformerModel(
            num_nodes=self.num_nodes,
            node_features=self.node_features,
            slide_win=self.slide_win,
            d_model=self.d_model,
            nhead=self.nhead,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        print(f'STformerWrapper — device: {self.device}, nodes: {self.num_nodes}, '
              f'features: {self.node_features}, d_model: {self.d_model}, '
              f'nhead: {self.nhead}, layers: {self.n_layers}')

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
        t0 = time.time()

        with torch.no_grad():
            for inputs, labels in tqdm(loader, total=len(loader), desc='Testing...'):
                y_hat = self.model(inputs)
                preds.append(y_hat.cpu().numpy())
                total_loss.append(self.loss_fn(y_hat, labels).item())
                errors.append((y_hat - labels).abs().cpu().numpy())

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
