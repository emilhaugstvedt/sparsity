import torch
from torch.utils.data import dataloader
import pickle
import os
import sys

sys.path.append("../")

from dynamic_sparse_reparameterization.DynamicSparseReparameterizationNet import DynamicSparseReparameterizationNet

# Load data
dataset = "alu" # alu or duffing

with open(f"../data/{dataset}/train.pickle", "rb") as f:
    train_data = pickle.load(f)

with open(f"../data/{dataset}/test.pickle", "rb") as f:
    test_data = pickle.load(f)

with open(f"../data/{dataset}/val.pickle", "rb") as f:
    val_data = pickle.load(f)

train_loader = dataloader.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = dataloader.DataLoader(test_data, batch_size=100, shuffle=True)
val_loader = dataloader.DataLoader(val_data, batch_size=100, shuffle=True)

### Hyperparamters ###
n_epochs = 100
lr_allocation = 0.01
lr_post_allocation = 0.001
weight_decay = 0 # L2 regulizer parameter for optimizer
epochs_with_reparameterization = 50
reallocation_frequency = 10
sparsity = 0.9

H = 0.01 # Initial thraeshold for reparameterization
Np = 100 # Number of reparameterizations per reallocation
fractional_tolerance = 0.1 # Fractional tolerance for adjustment of H during reparameterization

n_models = 10

#### Create model ####
input_size = train_data.x.shape[1]
output_size = train_data.y.shape[1]
layers = [input_size, 64, 64, 64, output_size]

for n in range(n_models):
    model = DynamicSparseReparameterizationNet(layers=layers, sparsity=sparsity, H=H, Np=Np, fractional_tolerence=fractional_tolerance)

    ### Train model ###            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_allocation)
    criterion = torch.nn.MSELoss()

    # Train before pruning, all weights are updated
    for epoch in range(epochs_with_reparameterization):
        for (x_batch, y_batch) in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
        
            # Mask the gradients to make sure the weights that are 0 stay 0
            for layer in model.layers:
                grad_mask = (layer.weight != 0)
                layer.weight.grad = torch.where(grad_mask, layer.weight.grad, torch.tensor(0.0))
            optimizer.step()

        if epoch <= epochs_with_reparameterization:
            if epoch % reallocation_frequency == 0 and epoch != 0:
                model.reallocate()
                print(f"Reallocated {model.K.item()} parameters\n")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {loss.item()}")
            print(f"Sparsity: {model.get_sparsity()} \n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_post_allocation)
    for epoch in range(epochs_with_reparameterization, n_epochs):
        for (x_batch, y_batch) in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
        
            # Mask the gradients to make sure the weights that are 0 stay 0
            for layer in model.layers:
                grad_mask = (layer.weight != 0)
                layer.weight.grad = torch.where(grad_mask, layer.weight.grad, torch.tensor(0.0))
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {loss.item()}")
            print(f"Sparsity: {model.get_sparsity()} \n")

    torch.save(model.state_dict(), f"../models/{dataset}/dynamic_sparse_reparameterization/model_{sparsity}_{n+1}.pickle")