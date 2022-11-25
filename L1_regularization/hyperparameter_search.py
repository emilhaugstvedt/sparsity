import torch
from torch.utils.data import dataloader
import pickle
import sys

sys.path.append("../")

import neptune.new as neptune

import os

from L1RegularizationNet import L1RegularizationNet

# Load data

dataset = "alu" # alu or duffing

with open(f"../data/{dataset}/{dataset}_train.pickle", "rb") as f:
    train_data = pickle.load(f)

with open(f"../data/{dataset}/{dataset}_test.pickle", "rb") as f:
    test_data = pickle.load(f)

with open(f"../data/{dataset}/{dataset}_val.pickle", "rb") as f:
    val_data = pickle.load(f)

train_loader = dataloader.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = dataloader.DataLoader(test_data, batch_size=100, shuffle=True)
val_loader = dataloader.DataLoader(val_data, batch_size=100, shuffle=True)

### Hyperparameters ###
n_epochs = 100
lr = 0.01
weight_decay = 0 # L2 regulizer parameter for optimizer
l1 = [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

#### Create model ####
input_size = train_data.x.shape[1]
output_size = train_data.y.shape[1]

layers = [input_size, 64, 64, 64, output_size]

loss_fn = torch.nn.MSELoss()
for l1_ in l1:

    model = L1RegularizationNet(layers=layers, l1=l1_)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Setup logging with Neptune
    run = neptune.init_run(
        project="emilhaugstvedt/l1-regularization",
        api_token=os.environ["NEPTUNE"],
    )

    # Log hyperparameters
    run["hyperparameters"] = {"n_epochs": n_epochs, "lr": lr, "l1":l1_}

    for epoch in range(n_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch) + model.l1 * model.L1_loss()
            loss.backward()
            optimizer.step()
    
        model.eval()
        with torch.no_grad():
            for (x_val, y_val) in val_loader:
                y_pred_val = model.forward(x_val)
                val_loss = loss_fn(y_pred_val, y_val)
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
    
    run.stop()