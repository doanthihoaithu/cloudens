import os.path

import torch
from torch.nn import GRU
from torch_geometric_temporal import A3TGCN2, ASTGCN
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class GRUCustom(torch.nn.Module):
    def __init__(self, num_nodes, node_features, hidden_dim, layer_dim, batch_size):
        super(GRUCustom, self).__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # Attention Temporal Graph Convolutional Cell
        self.rnn = GRU(input_size=num_nodes*node_features,
                       hidden_size=hidden_dim,
                       num_layers=layer_dim,
                       dropout=0.3,
                       batch_first=True)  # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(hidden_dim, num_nodes*node_features)

    def forward(self, x):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        batch_size, slide_win, num_nodes, node_feat = x.shape
        x = x.reshape(batch_size, slide_win, -1)

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        # edge_index = torch.LongTensor(np.array([range(num_nodes), range(num_nodes)])).to(x.device)
        # x = x.permute(0, 2, 1)
        out, _ = self.rnn(x, h0.detach())  # x [b, 207, 2, 12]  returns h [b, 207, 12]
        out = out[:, -1, :]
        # out = F.relu(out)
        out = self.linear(out)
        out = out.reshape(batch_size, num_nodes, node_feat)
        return out
class GRUWrapper:

    def __init__(self, num_nodes, node_features, hidden_dim, layer_dim, batch_size=32, device='cpu'):
        self.device = device
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.batch_size = batch_size
        self._init_model()


    def _init_model(self):
        # Making the model

        print(f'Device: {self.device}')
        model = GRUCustom(num_nodes=self.num_nodes, node_features=self.node_features,
                          hidden_dim=self.hidden_dim,
                          layer_dim=self.layer_dim,
                          batch_size=self.batch_size).to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.loss_fn = torch.nn.MSELoss()

        # print('Net\'s state_dict:')
        # total_param = 0
        # for param_tensor in model.state_dict():
        #     print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        #     total_param += np.prod(model.state_dict()[param_tensor].size())
        # print('Net\'s total params:', total_param)
        # # --------------------------------------------------
        # print('Optimizer\'s state_dict:')  # If you notice here the Attention is a trainable parameter
        # for var_name in self.optimizer.state_dict():
        #     print(var_name, '\t', self.optimizer.state_dict()[var_name])

        self.model = model

    def train(self, train_loader, val_loader, epochs):
        model = self.model

        train_losses = []
        valid_losses = []

        for epoch in range(epochs):
            model.train()
            step = 0
            loss_list = []
            for index, (encoder_inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                        desc=f'Training...'):
                # encoder_inputs = sample.x.to(DEVICE)
                # labels = sample.y.to(DEVICE)
                y_hat = model(encoder_inputs)  # Get model predictions
                loss = self.loss_fn(y_hat, labels)  # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                step = step + 1
                loss_list.append(loss.item())
                # if step % 100 == 0:
                #     print(sum(loss_list) / len(loss_list))
            epoch_train_loss = sum(loss_list) / len(loss_list)
            train_losses.append(epoch_train_loss)
            predictions, reconstruction_errors, epoch_valid_loss = self.predict(val_loader)
            valid_losses.append(epoch_valid_loss)

            print("Epoch {} train RMSE: {:.7f}, valid RMSE: {:.7f}".format(epoch, epoch_train_loss, epoch_valid_loss))

        history = {'epochs': epochs, 'train_losses': train_losses, 'valid_losses': valid_losses}
        self.history = history
        return history

    def predict(self, test_loader):
        self.model.eval()
        step = 0
        # Store for analysis
        total_loss = []
        batch_reconstruction_errors = []
        predictions = []
        for encoder_inputs, labels in tqdm(test_loader, total=len(test_loader), desc=f'Testing...'):
            # Get model predictions
            y_hat = self.model(encoder_inputs)
            predictions.append(y_hat.detach().cpu().numpy())
            # Mean squared error
            loss = self.loss_fn(y_hat, labels)
            total_loss.append(loss.item())
            batch_reconstruction_errors.append(abs(y_hat - labels).detach().cpu().numpy())

        reconstruction_errors = np.concatenate(batch_reconstruction_errors, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        return predictions, reconstruction_errors, sum(total_loss) / len(total_loss)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')


    def load(self, path):
        print(f'Model loaded from {path}')
        self.model.load_state_dict(torch.load(path))
