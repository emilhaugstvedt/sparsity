import torch
from torch.utils.data import dataloader
import sys
import pickle

sys.path.append("../")

import neptune.new as neptune

from L1RegularizationNet import L1RegularizationNet

from utils import load_data

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

### Hyperparameters ###
n_epochs = 100
lr = 0.01
weight_decay = 0 # L2 regulizer parameter for optimizer
l1 = 0.0005

n_models = 10

#### Create model ####
input_size = train_loader.dataset.x.shape[1]
output_size = train_loader.dataset.y.shape[1]

layers = [input_size, 64, 64, 64, output_size]

for n in range(n_models):

    model = L1RegularizationNet(layers=layers, l1=l1)

    ### Train model ###            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        for (x_batch, y_batch) in train_loader:
            model.train()
            optimizer.zero_grad()
            y_pred = model.forward(x_batch)
            loss = criterion(y_pred, y_batch)
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

    torch.save(model.state_dict(), f"../models/{dataset}/L1_regularization/model_{n+1}.pickle")