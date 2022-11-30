import torch
import torch.nn as nn
import torch.nn.functional as F

class PlainNet(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.n_inputs = layers[0]
        self.n_outputs = layers[-1]

        self.layers = nn.ModuleList()
        for l in range(len(layers[:-1])):
            self.layers.append(nn.Linear(in_features=layers[l], out_features=layers[l+1]))
            self.layers[l].weight.data.normal_(mean=0, std=0.5)
        

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def train_n_epochs(self,
                        train_loader,
                        n_epochs,
                        lr = 0.001,
                        loss_fn  = torch.nn.MSELoss(),
                        optimizer = "Adam",
                        verbose=False):
        
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        loss_fn = torch.nn.MSELoss()

        for epoch in range(n_epochs):
            for (x_batch, y_batch) in train_loader:
                optimizer.zero_grad()
                y_pred = self.forward(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch % 100 == 0) and verbose:
                print('Epoch {}: loss {}'.format(epoch, loss.item()))

    def get_sparsity(self):
        total_nonzero = 0
        total_n_weights = 0
        for layer in self.layers:
            total_nonzero = layer.weight.count_nonzero()
            total_n_weights = layer.weight.numel()
        return 1 - total_nonzero / total_n_weights
