import torch
from torch.utils.data import dataloader
import pickle
import os
import sys

sys.path.append("../")

from magnitude_pruning.MagnitudePruning import MagnitudePruningNet

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
lr = 0.01
weight_decay = 0.0001 # L2 regulizer parameter for optimizer
pruning_schedule = {"epoch": [25, 50, 75], "sparsity": [0.5, 0.5, 0.5]}
sparsity = 0.75

n_models = 10

#### Create model ####
input_size = train_data.x.shape[1]
output_size = train_data.y.shape[1]
layers = [input_size, 64, 64, 64, output_size]

for n in range(n_models):
    model = MagnitudePruningNet(layers=layers)

    ### Train model ###            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()


    # Train before pruning, all weights are updated
    for epoch in range(n_epochs):
        for (x_batch, y_batch) in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
        
            for layer in model.layers:
                grad_mask = (layer.weight != 0)
                layer.weight.grad = torch.where(grad_mask, layer.weight.grad, torch.tensor(0.0))
        
            optimizer.step()

        with torch.no_grad():
            for (x_batch, y_batch) in val_loader:
                y_pred = model.forward(x_batch)
                val_loss = criterion(y_pred, y_batch)
                break

        if (epoch in pruning_schedule["epoch"]):
            # Prune
            thresholds = model.get_weight_thresholds(sparsity=pruning_schedule["sparsity"][pruning_schedule["epoch"].index(epoch)])
            model.prune(thresholds)

        if (epoch % 10 == 0):
            print(f'Epoch {epoch}')
            print(f'Training loss {loss.item()}')
            print(f'Validation loss: {val_loss.item()}')
            print(f"Sparsity: {model.get_sparsity()}\n")


    torch.save(model.state_dict(), f"../models/{dataset}/magnitude_pruning/model_{n+1}.pickle")