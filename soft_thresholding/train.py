import torch
from torch.utils.data import dataloader
import pickle
import os
import sys

sys.path.append("../")

from soft_thresholding.SoftThresholdNet import SoftThresholdNet

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
n_epochs = 50
lr = 1e-3
l2 = 0 # L2 regulizer parameter for optimizer
l1 = 1e-4
s_init = 1 # initial value for threshold parameter
n_models = 10

#### Create model ####
input_size = train_data.x.shape[1]
output_size = train_data.y.shape[1]
layers = [input_size, 64, 64, 64, output_size]

for n in range(n_models):
    model = SoftThresholdNet(layers=layers, s_init=s_init)
    model.l1 = l1

    ### Train model ###            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    criterion = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        for (x_batch, y_batch) in train_loader:
            model.train()
            optimizer.zero_grad()
            y_pred = model.forward(x_batch)
            loss = criterion(y_pred, y_batch) + l1 * model.l1_loss()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for (x_val, y_val) in val_loader:
                y_pred_val = model.forward(x_val)
                val_loss = criterion(y_pred_val, y_val)
                break

        # Log train loss, validation loss and sparsity

        if (epoch % 10 == 0):
                print(f"Epoch {epoch}")
                print(f"Train loss: {loss}")
                print(f"Validation loss: {val_loss}")
                print(f"Sparsity: {model.get_sparsity()} \n")

    if l1 == 0:
        torch.save(model, f"../models/{dataset}/soft_thresholding/model_{n+1}.pickle")
    else:
        torch.save(model, f"../models/{dataset}/soft_thresholding/model_l1_{l1}_{n+1}.pickle")