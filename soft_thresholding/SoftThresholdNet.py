import torch
import torch.nn as nn
from torch.nn import functional as F

def sparse_function(x, s, activation=torch.relu, f=torch.sigmoid):
    return torch.sign(x) * activation(torch.abs(x) - f(s))

class SoftThresholdLayer(nn.Module):
    def __init__(self, in_features, out_features, s_init=None):
        super().__init__()

        self.l1 = None

        self.in_features = in_features
        self.out_features = out_features

        self.n_weights = in_features * out_features

        weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(weight)
        nn.init.normal_(self.weight)

        s = torch.Tensor(1, 1)
        self.s = nn.Parameter(s)

        if s_init:
            nn.init.constant_(self.s, s_init)
        else:
            nn.init.zeros_(self.s)

    def forward(self, x):
        sparse_weight = sparse_function(self.weight, self.s)
        return torch.matmul(x, sparse_weight)
    
    def get_sparse_weights(self):
        return sparse_function(self.weight, self.s)

    def get_sparsity(self):
        sparse_weight = sparse_function(self.weight, self.s)
        return (1 - torch.count_nonzero(sparse_weight) / self.n_weights)


class SoftThresholdNet(nn.Module):
    def __init__(self, layers, s_init=None):
        super().__init__()

        self.n_inputs = layers[0]
        self.n_outputs = layers[-1]

        self.layers = nn.ModuleList()
        for l in range(len(layers[:-1])):
            self.layers.append(SoftThresholdLayer(layers[l], layers[l+1], s_init=s_init))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


    def train_n_epochs(self,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    n_epochs: int,
                    lr = 0.001,
                    weight_decay=0,
                    loss_fn  = torch.nn.MSELoss(),
                    optimizer = "Adam",
                    verbose=False):
        
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        loss_fn = torch.nn.MSELoss()

        for epoch in range(n_epochs):
            for (x_batch, y_batch) in train_loader:
                self.train()
                optimizer.zero_grad()
                y_pred = self.forward(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch % 10 == 0) and verbose:
                self.eval()
                with torch.no_grad():
                    for (x_val, y_val) in val_loader:
                        y_pred_val = self.forward(x_val)
                        val_loss = loss_fn(y_pred_val, y_val)
                        break
                print(f"Epoch {epoch}")
                print(f"Train loss: {loss}")
                print(f"Validation loss: {val_loss}")
                print(f"Sparsity: {self.get_sparsity()} \n")

    def get_sparsity(self):
        non_zero_weights = 0
        total_weights = 0

        for layer in self.layers:
            non_zero_weights += torch.count_nonzero(layer.get_sparse_weights())
            total_weights += layer.in_features * layer.out_features
        
        return (1 - non_zero_weights / total_weights).item()
    
    def l1_loss(self):
        return sum([layer.weight.abs().sum() for layer in self.layers])
        