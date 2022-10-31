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

        self.layers = nn.ModuleList()
        for layer in hidden_sizes:
            self.layers.append(nn.Linear(layer[0], layer[1]))
        
        self.output = nn.Linear(hidden_sizes[-1][-1], n_outputs)

    def forward(self, x):
    
        x = F.relu(self.input(x))

        for layer in self.layers:
            x = F.relu(layer(x))
    
        x = self.output(x)

        return x

    def train_n_epochs(self, training_loader, n_epochs, lr = 0.001, verbose=False):
        optimizer =  torch.optim.Adam(self.parameters(), lr=0.001)
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

