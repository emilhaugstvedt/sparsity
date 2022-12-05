import torch
import torch.nn as nn
import numpy as np

class MagnitudePruningNet(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.n_inputs = layers[0]
        self.n_outputs = layers[-1]

        self.n_weights = sum([layers[i] * layers[i + 1] for i in range(len(layers) - 1)])

        self.layers = nn.ModuleList()
        for l in range(len(layers[:-1])):
            self.layers.append(nn.Linear(in_features=layers[l], out_features=layers[l+1]))
            weight = torch.Tensor(layers[l+1], layers[l])
            self.layers[l].weight = nn.Parameter(weight)
            nn.init.kaiming_normal_(self.layers[l].weight)

        self.pruning_shedule = None
        

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def prune(self, thresholds):
        for layer, threshold in zip(self.layers, thresholds):
            layer.weight = nn.Parameter(torch.where(torch.abs(layer.weight) < threshold, torch.tensor(0.0), layer.weight))

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.reshape(-1))
        return weights

    def get_weight_thresholds(self, sparsity):
        thresholds = []
        for layer in self.layers:
            weights = layer.weight
            weights = weights.reshape(-1)
            weights = torch.abs(weights)
            weights = torch.sort(weights)[0]
            threshold = weights[int((len(weights) + (weights == 0).count_nonzero()) * sparsity)]
            thresholds.append(threshold)
        return thresholds
        
    def train_n_epochs(self, 
                        train_loader,
                        val_loader,
                        n_epochs, 
                        pruning_epoch=None, 
                        sparsity=None,
                        lr = 0.001,
                        weight_decay = 0.0,
                        loss_fn  = torch.nn.MSELoss(),
                        optimizer = "Adam",
                        verbose=False):

        if optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        loss_fn = torch.nn.MSELoss()

        # pruning_epoch = Nonde -> No pruning and only non-zero weights are trained
        if pruning_epoch != None:

            # Train before pruning, all weights are updated
            for epoch in range(pruning_epoch):
                for (x_batch, y_batch) in train_loader:
                    optimizer.zero_grad()
                    y_pred = self.forward(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                
                with torch.no_grad():
                    for (x_batch, y_batch) in val_loader:
                        y_pred = self.forward(x_batch)
                        val_loss = loss_fn(y_pred, y_batch)
                        break

                if (epoch % 10 == 0) and verbose:
                    print(f'Epoch {epoch}')
                    print(f'Training loss {loss.item()}')
                    print(f'Validation loss: {val_loss.item()}')

            # Prune
            thresholds = self.get_weight_thresholds(sparsity)
            self.prune(thresholds)

            if verbose:
                print(f"Pruned. Sparsity is now {self.get_sparsity()}")
        else:
            pruning_epoch = 0
        
        # Train after pruning, now only weights that are not pruned are updated
        for epoch in range(pruning_epoch, n_epochs):
            for (x_batch, y_batch) in train_loader:
                optimizer.zero_grad()
                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()

                # Mask the gradients to make sure the weights that are 0 stay 0
                for layer in self.layers:
                    grad_mask = (layer.weight != 0)
                    layer.weight.grad = torch.where(grad_mask, layer.weight.grad, torch.tensor(0.0))

                optimizer.step()

            with torch.no_grad():
                for (x_batch, y_batch) in val_loader:
                    y_pred = self.forward(x_batch)
                    val_loss = loss_fn(y_pred, y_batch)
                    break

            if (epoch % 10 == 0) and verbose:
                print(f'Epoch {epoch}')
                print(f'Training loss {loss.item()}')
                print(f'Validation loss: {val_loss.item()}')

            

    def get_sparsity(self):
       return (1 - (torch.tensor([layer.weight.count_nonzero() for layer in self.layers]).sum()/self.n_weights)).detach()

    def get_layerwise_sparsity(self):
        return [(1 - (layer.weight.count_nonzero()/(layer.weight.shape[0]  * layer.weight.shape[1]))).detach().numpy() for layer in self.layers]

    def l1_loss(self):
        return sum([layer.weight.abs().sum() for layer in self.layers])