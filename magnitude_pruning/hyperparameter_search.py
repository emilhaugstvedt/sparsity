import torch
from torch.utils.data import dataloader
import pickle
import os
import sys
import numpy as np

sys.path.append("../")

import neptune.new as neptune

from magnitude_pruning.MagnitudePruning import MagnitudePruningNet
from utils import load_data

# Load data
dataset = "alu" # alu or duffing

train_loader, test_loader, val_loader = load_data(f"../data/{dataset}")

### Hyperparamters ###
n_epochs = 200
lr_pre_pruning = 0.01
lr_post_pruning = 0.0001
weight_decay = 0 # L2-regularization parameter for optimizer
sparsities = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
pruning_epoch = 100


### Create model ###
input_size = train_loader.dataset.x.shape[1]
output_size = train_loader.dataset.y.shape[1]

layers = [input_size, 64, 64, 64, output_size]

### Loss function ###
loss_fn = torch.nn.MSELoss()

# Train before pruning, all weights are updated
for sparsity in sparsities:

    run = neptune.init_run(
        project="emilhaugstvedt/magnitude-pruning",
        api_token=os.environ["NEPTUNE"],
    )

    run["hyperparameters"] = {"n_epochs": n_epochs, "lr pre pruning": lr_pre_pruning, "lr post pruning": lr_post_pruning, "sparsity":sparsity}

    model = MagnitudePruningNet(layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_pre_pruning)

    for epoch in range(pruning_epoch):
        for (x_batch, y_batch) in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            for (x_batch, y_batch) in val_loader:
                y_pred = model.forward(x_batch)
                val_loss = loss_fn(y_pred, y_batch)
                break

        if (epoch % 10 == 0):
            print(f'Epoch {epoch}')
            print(f'Training loss {loss.item()}')
            print(f'Validation loss: {val_loss.item()}')
        
        # Log train loss, validation loss and sparsity
        run["validation/loss"].log(val_loss)
        run["train/loss"].log(loss)
        run["train/sparsity"].log(model.get_sparsity())

    # Prune
    thresholds = model.get_weight_thresholds(sparsity)
    model.prune(thresholds)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_post_pruning)

    # Train after pruning, now only weights that are not pruned are updated
    for epoch in range(pruning_epoch, n_epochs):
        for (x_batch, y_batch) in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()

            # Mask the gradients to make sure the weights that are 0 stay 0
            for layer in model.layers:
                grad_mask = (layer.weight != 0)
                layer.weight.grad = torch.where(grad_mask, layer.weight.grad, torch.tensor(0.0))

            optimizer.step()

        with torch.no_grad():
            for (x_batch, y_batch) in val_loader:
                y_pred = model.forward(x_batch)
                val_loss = loss_fn(y_pred, y_batch)
                break

        if (epoch % 10 == 0):
            print(f'Epoch {epoch}')
            print(f'Training loss {loss.item()}')
            print(f'Validation loss: {val_loss.item()}')
        
        # Log train loss, validation loss and sparsity
        run["validation/loss"].log(val_loss)
        run["train/loss"].log(loss)
        run["train/sparsity"].log(model.get_sparsity())
    
    run.stop()
