import torch
import torch.nn as nn

class DynamicReparameterizationLayer(nn.Module):
    def __init__(self, n_input, n_output, sparsity) -> None:
        super().__init__()
        
        self.n_weights = n_input * n_output 

        weight = torch.Tensor(n_input, n_output)
        self.weight = nn.Parameter(weight)
        nn.init.sparse_(self.weight, sparsity=sparsity) # Legge til std-parameter?

        self.m = self.get_number_of_nonzero_weights() # Store number of nonzero weights in layer

        self.r = 0 # Store number of zero weights in layer
        self.k = 0 # Number of pruned weights in last pruning step

    def forward(self, x):
        return torch.matmul(x, self.weight)
        

    def prune(self, H) -> int:
        k = len((abs(self.weight) < H).nonzero())
        self.weight = nn.Parameter(torch.where(abs(self.weight) < H, 0, self.weight))

        self.k = k
        r = self.m - k
        self.r = r

        return k, r
    
    def grow(self, K, R):
        g = int((self.r / R) * K)

        with torch.no_grad():
            for _ in range(g):
                zero_indices = (self.weight == 0).nonzero()
                if len(zero_indices) == 0:
                    break
                random_index = zero_indices[torch.randint(0, len(zero_indices), (1,))][0]
                self.weight[random_index[0]][random_index[1]] = torch.normal(mean=0, std=0.01, size=(1,))

        self.m = self.get_number_of_nonzero_weights()
        
    def get_number_of_nonzero_weights(self) -> int:
        return torch.count_nonzero(self.weight)

    def get_sparsity(self):
        return (1 - self.m / self.n_weights)

class DynamicReparameterizationNet(nn.Module):
    def __init__(self, n_inputs, hidden_layers, n_outputs, H, sparsity, Np, fractional_tolerence):
        super().__init__()

        self.n_inputs = n_inputs
        self.hidden_sizes = hidden_layers
        self.n_outputs = n_outputs

        # Percentage of weights that should be 0
        self.sparsity = sparsity

        # Adaptive threshold
        self.H = H

        # Tolerence for adjusting H
        self.fractional_tolerance = fractional_tolerence
        self.Np = Np

        self.layers = nn.ModuleList()
        self.layers.append(DynamicReparameterizationLayer(n_input=n_inputs, n_output=hidden_layers[0][0], sparsity=sparsity))
        for layer in hidden_layers:
            self.layers.append(DynamicReparameterizationLayer(n_input=layer[0], n_output=layer[1], sparsity=sparsity))
        self.layers.append(DynamicReparameterizationLayer(n_input=hidden_layers[-1][1], n_output=n_outputs, sparsity=sparsity))

        # Number of nonzero weights in the network
        self.M = sum([layer.get_number_of_nonzero_weights() for layer in self.layers])

        self.n_weights = sum([layer.n_weights for layer in self.layers])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def reallocate(self):
        #print("Reallocation starting, current sparsity: ", self.get_sparsity())
        K , M = self.prune(self.H)
        #print("Pruning done, pruned ", K, " weights")
        self.adjust_H(K, self.Np)
        #print(f"H adjusted to {self.H}")
        #print(K)
        self.grow(K, M)
        #print("Reallocation done, new sparsity: ", self.get_sparsity())
        self.M = sum([layer.get_number_of_nonzero_weights() for layer in self.layers])
        #print(sum([layer.n_weights for layer in self.layers]))

    def adjust_H(self, K, Np) -> None:
        if K < (1 - self.fractional_tolerance) * Np:
            self.H = 2 * self.H
            return
        elif K > (1 + self.fractional_tolerance) * Np:
            self.H = 1/2 * self.H
            return
        return

    def grow(self, K, R):
        for layer in self.layers:
            layer.grow(K, R)

    def prune(self, H):
        K = R = 0
        for layer in self.layers:
            k, r = layer.prune(self.H)
            K += k
            R += r
        return K, R

    def train_n_epochs(
                    self,
                    train_loader,
                    n_epochs,
                    lr,
                    weight_decay,
                    epochs_reallocate,
                    optimizer="Adam",
                    criterion="MSE"):

            
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            print("Optimizer not supported")
            return
        
        if criterion == "MSE":
            criterion = nn.MSELoss()
        else:
            print("Criterion not supported")
            return

        for epoch in range(n_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = self(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: {loss.item()}")
                print(f"Sparsity: {self.get_sparsity()} \n")
            
            if epoch % epochs_reallocate == 0 and epoch != 0:
                self.reallocate()

    def get_sparsity(self) -> int:
        return (1.0 - self.M/self.n_weights).detach().numpy()

    def get_layerwise_sparsity(self):
        return [layer.get_sparsity() for layer in self.layers]