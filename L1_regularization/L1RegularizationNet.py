import torch
import sys

sys.path.append("../")

from plain_net.PlainNet import PlainNet

class L1RegularizationNet(PlainNet):
    def __init__(self, layers):
        super().__init__(layers)

    def l1_loss(self):
        return sum([layer.weight.abs().sum() for layer in self.layers])

    def train_n_epochs(self, train_loader, n_epochs, lr=0.001, l1 = 0.001, loss_fn=..., optimizer="Adam", verbose=False):
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        loss_fn = torch.nn.MSELoss()

        for epoch in range(n_epochs):
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch) + l1 * self.l1_loss()
                loss.backward()
                optimizer.step()

            if epoch % 100 == 0 and verbose:
                print(f"Epoch {epoch}: {loss}")
                print(f"Sparsity: {self.get_sparsity()}")