import torch
import torch.nn as nn
import torch.nn.functional as F

class plainNet(torch.nn.Module):
    def __init__(self, n_inputs, hidden_sizes, n_outputs):
        super(plainNet, self).__init__()

        self.n_inputs = n_inputs
        self.hidden_sizes = hidden_sizes
        self.n_outputs = n_outputs

        self.input = nn.Linear(n_inputs, hidden_sizes[0][0])

        self.layers = []
        for size in hidden_sizes:
            self.layers.append(nn.Linear(size[0], size[1]))
        
        self.output = nn.Linear(hidden_sizes[-1][-1], n_outputs)

    def forward(self, x):
    
        x = F.relu(self.input(x))

        for layer in self.layers:
            x = F.relu(layer(x))
    
        x = F.relu(self.output(x))

        return x

    def train_n_epochs(n_epochs):

        