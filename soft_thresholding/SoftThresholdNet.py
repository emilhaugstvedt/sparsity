from tkinter.messagebox import NO
import torch
import torch.nn as nn
from torch.nn import functional as F

def sparse_function(x, s, activation=torch.relu, f=torch.sigmoid):
    return torch.sign(x) * activation(torch.abs(x) - f(s))

class SoftThresholdLayer(nn.Module):
    def __init__(self, in_features, out_features, s_init=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(weight)
        nn.init.normal_(self.weight)

        # S kan endres til å være global
        # Kan også ha en s for hver vekt

        s = torch.Tensor(1, 1)
        self.s = nn.Parameter(s)

        if s_init:
            nn.init.constant_(self.s, s_init)
        else:
            nn.init.zeros_(self.s)

    def forward(self, x):

        sparse_weight = sparse_function(
            self.weight,
            self.s
        )

        return torch.matmul(x, sparse_weight)
    
    def get_sparse_weights(self):
        return sparse_function(self.weight, self.s)


class SoftThresholdNet(nn.Module):
    def __init__(self, n_inputs, hidden_sizes, n_outputs, s_init=None):
        super().__init__()

        self.n_inputs = n_inputs
        self.hidden_sizes = hidden_sizes
        self.n_outputs = n_outputs

        self.layers = nn.ModuleList()
        
        input = SoftThresholdLayer(n_inputs, hidden_sizes[0][0], s_init=s_init)
        self.layers.append(input)

        for layer in hidden_sizes:
            self.layers.append(SoftThresholdLayer(layer[0], layer[1], s_init=s_init))

        output = SoftThresholdLayer(hidden_sizes[-1][-1], n_outputs, s_init=s_init)
        self.layers.append(output)

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
    
        x = self.layers[-1](x)

        return x

    def train_n_epochs(self,
                    training_loader,
                    n_epochs,
                    lr = 0.001,
                    weight_decay=0,
                    loss_fn  = torch.nn.MSELoss(),
                    optimizer = "Adam",
                    verbose=False):
        
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        loss_fn = torch.nn.MSELoss()

        for epoch in range(n_epochs):
            for (x_batch, y_batch) in training_loader:
                optimizer.zero_grad()
                y_pred = self.forward(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch % 100 == 0) and verbose:
                print('Epoch {}: loss {}'.format(epoch, loss.item()))

    def get_sparsity(self):
        
        non_zero_weights = 0
        total_weights = 0

        for layer in self.layers:
            non_zero_weights += torch.count_nonzero(layer.get_sparse_weights())
            total_weights += layer.in_features * layer.out_features
        
        return (total_weights - non_zero_weights) / total_weights
        