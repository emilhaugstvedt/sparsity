import torch
from torch.utils.data import dataloader
import pickle

import neptune.new as neptune

import os

from dynamic_sparse_reparameterization.DynamicSparseReparameterizationNet import DynamicSparseReparameterizationNet

run = neptune.init_run(
    project="emilhaugstvedt/dynamic-sparse-reparameterization",
    api_token=os.environ["NEPTUNE"],
)

# Load data

dataset = "alu" # alu or duffing

with open(f"data/{dataset}/{dataset}_train.pickle", "rb") as f:
    train_data = pickle.load(f)

with open(f"data/{dataset}/{dataset}_test.pickle", "rb") as f:
    test_data = pickle.load(f)

with open(f"data/{dataset}/{dataset}_val.pickle", "rb") as f:
    val_data = pickle.load(f)

train_loader = dataloader.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = dataloader.DataLoader(test_data, batch_size=100, shuffle=True)
val_loader = dataloader.DataLoader(val_data, batch_size=100, shuffle=True)

### Hyperparameters ###
n_epochs = 500
lr = 0.001
weight_decay = 0 # L2 regulizer parameter for optimizer

H = 0.001
sparsity= 0.9
Np = [10, 100, 1000, 10000]
fractional_tolerence = 0.1
epochs_reallocate = [5, 10, 15, 20, 25, 50, 75, 100, 150, 300]

#### Create model ####
input_size = train_data.x.shape[1]
output_size = train_data.y.shape[1]

layers = [input_size, 64, 64, 64, output_size]

for Np_ in Np:

    # Log hyperparameters
    run["hyperparameters"] = {"n_epochs": n_epochs, "lr": lr, "weight_decay": weight_decay, "H": H, "sparsity": sparsity, "Np": Np_, "fractional_tolerence": fractional_tolerence, "epochs_reallocate": epochs_reallocate}

    model = DynamicSparseReparameterizationNet(layers=layers, H=H, sparsity=sparsity, Np=Np_, fractional_tolerence=fractional_tolerence)

    ### Train model ###            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            model.train()
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            # Mask the gradients to make sure the weights that are 0 stay 0
            for layer in model.layers:
                grad_mask = (layer.weight != 0)
                layer.weight.grad = torch.where(grad_mask, layer.weight.grad, torch.tensor(0.0))

            optimizer.step()
                            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: {loss.item()}")
            print(f"Sparsity: {model.get_sparsity()} \n")
                
        if epoch in epochs_reallocate:
                model.reallocate()
        
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

    #params = {"learning_rate": 0.001, "optimizer": "Adam"}
    run["parameters"] = model.parameters()

    run.stop()