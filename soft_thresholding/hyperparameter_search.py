import torch
from torch.utils.data import dataloader
import pickle
import os
import sys

sys.path.append("../")

import neptune.new as neptune

from soft_thresholding.SoftThresholdNet import SoftThresholdNet

# Load data
dataset = "alu" # alu or duffing

with open(f"../data/{dataset}/{dataset}_train.pickle", "rb") as f:
    train_data = pickle.load(f)

with open(f"../data/{dataset}/{dataset}_test.pickle", "rb") as f:
    test_data = pickle.load(f)

with open(f"../data/{dataset}/{dataset}_val.pickle", "rb") as f:
    val_data = pickle.load(f)

train_loader = dataloader.DataLoader(train_data, batch_size=50, shuffle=True)
test_loader = dataloader.DataLoader(test_data, batch_size=100, shuffle=True)
val_loader = dataloader.DataLoader(val_data, batch_size=100, shuffle=True)

### Hyperparamters ###
n_epochs = 100
lr = 0.01
weight_decay = [0, 0.1, 0.01, 0.001, 0.0001] # L2 regulizer parameter for optimizer
s_init = 1 # initial value for threshold parameter

### - ###

for weight_decay_ in weight_decay:

    # Init logging on neptune
    run = neptune.init_run(
        project="emilhaugstvedt/soft-thresholding",
        api_token=os.environ["NEPTUNE"],
    )
    
    # Log hyperparameters
    run["hyperparameters"] = {"n_epochs": n_epochs, "lr": lr, "weight_decay": weight_decay_, "s_init": s_init}


    #### Create model ####
    input_size = train_data.x.shape[1]
    output_size = train_data.y.shape[1]

    layers = [input_size, 64, 64, 64, output_size]

    model = SoftThresholdNet(layers=layers, s_init=s_init)

    ### Train model ###            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_)

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
        run["validation/loss"].log(val_loss)
        run["train/loss"].log(loss)
        run["train/sparsity"].log(model.get_sparsity())

        if (epoch % 10 == 0):
                print(f"Epoch {epoch}")
                print(f"Train loss: {loss}")
                print(f"Validation loss: {val_loss}")
                print(f"Sparsity: {model.get_sparsity()} \n")

    run["parameters"].log(model.parameters())

    run.stop()