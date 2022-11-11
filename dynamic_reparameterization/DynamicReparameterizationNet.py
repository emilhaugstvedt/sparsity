import torch
import torch.nn as nn

class DynamicReparameterizationLayer(nn.Module):
    def __init__(self, n_input, n_output, sparsity) -> None:
        super().__init__()
        
        self.n_weights = n_input * n_output 

        self.sparsity = sparsity

        weight = torch.Tensor(n_input, n_output)
        self.weight = nn.Parameter(weight)
        # Custom sparsity initialization
        nn.init.zeros_(self.weight)
        with torch.no_grad():
            for _ in range(int(self.n_weights * (1-self.sparsity))):
                indices = (self.weight == 0).nonzero() 
                random_index = indices[torch.randint(0, len(indices), (1,))][0]
                self.weight[random_index[0]][random_index[1]] = torch.rand(1)

        self.m = self.get_number_of_nonzero_weights() # Store number of nonzero weights in layer

        self.r = 0 # Store number of zero weights in layer
        self.k = 0 # Number of pruned weights in last pruning step
            

    def forward(self, x):
        return torch.matmul(x, self.weight)
        

    def prune(self, H) -> int:
        k = torch.mul((self.weight < H), (self.weight != 0)).count_nonzero()
        with torch.no_grad():
            self.weight = nn.Parameter(torch.where(((self.weight.abs() < H) & (self.weight != torch.tensor(0))), 0, self.weight))
        self.k = k
        r = self.m - k
        self.r = r

        return self.k, self.r
    
    def grow(self, K, R):
        if R != 0:
            g = int((self.r / R) * K)
        else:
            g = self.k

        with torch.no_grad():
            for _ in range(g):
                zero_indices = (self.weight == 0).nonzero()
                if len(zero_indices) == 0:
                    break
                random_index = zero_indices[torch.randint(0, len(zero_indices), (1,))][0]
                self.weight[random_index[0]][random_index[1]] = torch.rand(1)
        self.m = self.get_number_of_nonzero_weights()
        
        return g

    def get_number_of_nonzero_weights(self) -> int:
        return torch.count_nonzero(self.weight)

    def get_sparsity(self):
        return (1 - (self.weight.count_nonzero() / self.n_weights))

class DynamicReparameterizationNet(nn.Module):
    def __init__(self, layers, H, sparsity, Np, fractional_tolerence, verbose=False) -> None:
        super().__init__()

        self.verbose = verbose

        self.n_inputs = layers[0]
        self.hidden_sizes = len(layers) - 2
        self.n_outputs = layers[-1]

        # Percentage of weights that should be 0
        self.sparsity = sparsity

        # Adaptive threshold
        self.H = H

        # Tolerence for adjusting H
        self.fractional_tolerance = fractional_tolerence
        self.Np = Np

        self.layers = nn.ModuleList()
        for l in range(len(layers[:-1])):
            self.layers.append(DynamicReparameterizationLayer(n_input=layers[l], n_output=layers[l+1], sparsity=sparsity))

        # Number of nonzero weights in the network
        self.M = sum([layer.get_number_of_nonzero_weights() for layer in self.layers])

        self.n_weights = sum([layer.n_weights for layer in self.layers])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def reallocate(self):

        K , R = self.prune(self.H)
        if self.verbose:
            print(f"Pruned {K.item()} weights")

        self.adjust_H(K, self.Np)
        
        if self.verbose:
            print(f"Adjusted H to {self.H}")

        G = self.grow(K, R)
        
        if self.verbose:
            print(f"Grew {G} weights")

        self.M = sum([layer.get_number_of_nonzero_weights() for layer in self.layers])

    def adjust_H(self, K, Np) -> None:
        if K < (1 - self.fractional_tolerance) * Np :
            self.H = 2 * self.H
            return
        elif K > (1 + self.fractional_tolerance) * Np:
            self.H = 1/2 * self.H
            return
        #else:
            #self.H = 2/3 * self.H
        return

    def grow(self, K, R):
        G = 0
        for layer in self.layers:
            g = layer.grow(K, R)
            G += g
        return G

    def prune(self, H):
        K = R = 0
        for layer in self.layers:
            k, r = layer.prune(H)
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

                # Mask the gradients to make sure the weights that are 0 stay 0
                for layer in self.layers:
                    grad_mask = (layer.weight != 0)
                    layer.weight.grad = torch.where(grad_mask, layer.weight.grad, torch.tensor(0.0))

                optimizer.step()
                        
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: {loss.item()}")
                print(f"Sparsity: {self.get_sparsity()} \n")
            
            if epochs_reallocate != 0:
                if epoch % epochs_reallocate == 0 and epoch != 0:
                    self.reallocate()

    def get_sparsity(self) -> int:
        return 1.0 - (torch.tensor([layer.get_number_of_nonzero_weights() for layer in self.layers]).sum()/self.n_weights).detach().numpy()

    def get_layerwise_sparsity(self):
        return [layer.get_sparsity() for layer in self.layers]