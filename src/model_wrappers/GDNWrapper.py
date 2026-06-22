import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax as _scatter_softmax
from tqdm import tqdm


def _batched_scatter_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax over incoming edges per destination node, applied independently per batch item."""
    B, E = scores.shape
    flat = scores.reshape(-1)
    # Unique index per (batch item, destination node) so softmax groups stay within batch
    batch_offset = torch.arange(B, device=scores.device).unsqueeze(1).expand(-1, E).reshape(-1) * num_nodes
    idx = dst.unsqueeze(0).expand(B, -1).reshape(-1) + batch_offset
    return _scatter_softmax(flat, idx).reshape(B, E)


class GDNModel(nn.Module):
    """
    Graph Deviation Network.
    Learns a sparse graph from trainable node embeddings via top-k cosine
    similarity, then uses graph attention over GRU-encoded node histories
    to forecast the next time step.

    Reference: Deng & Hooi, "Graph Neural Network-Based Anomaly Detection
               in Multivariate Time Series", AAAI 2021.
    """

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 embed_dim: int = 64, hidden_dim: int = 64, topk: int = 20):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.topk = min(topk, num_nodes - 1)

        # Learnable node embeddings — drive graph structure learning
        self.node_embedding = nn.Embedding(num_nodes, embed_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)

        # Temporal encoder: one shared GRU processes every node's history
        self.gru = nn.GRU(node_features, hidden_dim, batch_first=True)

        # Graph attention projections
        self.W_h = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.W_e = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_v = nn.Linear(2 * embed_dim, 1, bias=False)

        # Forecasting head: aggregated state → next-step prediction
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_features),
            nn.Sigmoid(),
        )

    def _topk_edges(self):
        """Top-k directed edges via cosine similarity of node embeddings (neighbour → node)."""
        emb = self.node_embedding.weight                     # [N, d]
        sim = F.normalize(emb, dim=-1) @ F.normalize(emb, dim=-1).T  # [N, N]
        sim.fill_diagonal_(float('-inf'))
        _, idx = sim.topk(self.topk, dim=1)                 # [N, k]
        dst = torch.arange(self.num_nodes, device=emb.device) \
                   .unsqueeze(1).expand(-1, self.topk).reshape(-1)
        src = idx.reshape(-1)                               # src is the neighbour
        return src, dst

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F]
        B, T, N, F_in = x.shape

        # 1. Encode each node's temporal history with a shared GRU
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, F_in)
        _, h_n = self.gru(x_flat)                           # [1, B*N, hidden]
        h = h_n.squeeze(0).reshape(B, N, -1)                # [B, N, hidden]

        # 2. Build sparse top-k graph from learnable embeddings
        src, dst = self._topk_edges()                       # each length E = N*topk

        # 3. Compute attention scores for each edge
        emb = self.node_embedding.weight                    # [N, d]
        attn_in = torch.cat([
            self.W_h(h[:, dst, :]),                         # target hidden state [B, E, d]
            self.W_e(emb[src]).unsqueeze(0).expand(B, -1, -1),  # source embedding [B, E, d]
        ], dim=-1)                                          # [B, E, 2d]
        scores = self.attn_v(F.leaky_relu(attn_in, 0.2)).squeeze(-1)   # [B, E]
        alpha = _batched_scatter_softmax(scores, dst, N)                # [B, E]

        # 4. Aggregate neighbour hidden states into each destination node
        agg = torch.zeros(B, N, h.shape[-1], device=x.device)
        agg.scatter_add_(
            1,
            dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, h.shape[-1]),
            alpha.unsqueeze(-1) * h[:, src, :],             # [B, E, hidden]
        )

        # 5. Predict next time step from aggregated state + node embedding
        feat = torch.cat([agg, emb.unsqueeze(0).expand(B, -1, -1)], dim=-1)  # [B, N, hidden+d]
        return self.out_layer(feat)                         # [B, N, F]


class GDNWrapper:

    def __init__(self, num_nodes: int, node_features: int, slide_win: int,
                 embed_dim: int = 64, hidden_dim: int = 64, topk: int = 20,
                 batch_size: int = 32, device='cpu'):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.slide_win = slide_win
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.batch_size = batch_size
        self.device = device
        self.inference_time = 0
        self._init_model()

    def _init_model(self):
        self.model = GDNModel(
            num_nodes=self.num_nodes,
            node_features=self.node_features,
            slide_win=self.slide_win,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            topk=self.topk,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        print(f'GDNWrapper — device: {self.device}, nodes: {self.num_nodes}, '
              f'features: {self.node_features}, topk: {self.model.topk}')

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