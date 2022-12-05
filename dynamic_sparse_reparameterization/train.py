import torch
from torch.utils.data import dataloader
import pickle
import os
import sys
from tqdm import tqdm
import copy

import neptune.new as neptune

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

train_loader = dataloader.DataLoader(train_data, batch_size=50, shuffle=True)
test_loader = dataloader.DataLoader(test_data, batch_size=100, shuffle=True)
val_loader = dataloader.DataLoader(val_data, batch_size=100, shuffle=True)

### Hyperparamters ###
n_epochs = 200
lr = 1e-3
l2 = 0
l1 = 1e-4
reparameterization_start_epoch = 0
reparameterization_end_epoch = 100
reallocation_frequency = 25
sparsity = 0.1

scheduler_step_size = 10
scheduler_gamma = 0.1
scheduler_start_epoch = 100

swa_start_epoch = 100

H = 0.001 # Initial thraeshold for reparameterization
Np = 150 # Number of reparameterizations per reallocation
fractional_tolerance = 0.1 # Fractional tolerance for adjustment of H during reparameterization

n_models = 10

#### Create model ####
input_size = train_data.x.shape[1]
output_size = train_data.y.shape[1]
layers = [input_size, 128, 128, 128, 128, output_size]

ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
             0.2 * averaged_model_parameter + 0.8 * model_parameter

for n in range(n_models):

    print(f"### Traning model {n+1} ###")

    run = neptune.init_run(
        project="dynamic-sparse-reparameterization",
        name=f"Model {n+1}",
        api_token=os.environ["NEPTUNE"]
    )

    run["hyperparameters"] = {
        "n_epochs": n_epochs,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "reparameterization_start_epoch": reparameterization_start_epoch,
        "reparameterization_end_epoch": reparameterization_end_epoch,
        "reallocation_frequency": reallocation_frequency,
        "sparsity": sparsity,
        "H": H,
        "Np": Np,
        "fractional_tolerance": fractional_tolerance
    }

    model = DynamicSparseReparameterizationNet(layers=layers, sparsity=sparsity, H=H, Np=Np, fractional_tolerence=fractional_tolerance)
    swa_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)

    lowest_val_loss = 99999
    best_model = model

    lowest_swa_val_loss = 99999
    best_swa_model = swa_model

    ### Train model ###            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Train before pruning, all weights are updated
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for (x_batch, y_batch) in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(x_batch)
            swa_train_loss = criterion(swa_model(x_batch), y_batch)
            loss = criterion(y_pred, y_batch) + l1 * model.l1_loss()
            loss.backward()

            # Mask the gradients to make sure the weights that are 0 stay 0
            for layer in model.layers:
                grad_mask = (layer.weight != 0)
                layer.weight.grad = torch.where(grad_mask, layer.weight.grad, torch.tensor(0.0))
            optimizer.step()

        if epoch > scheduler_start_epoch:
            scheduler.step()

        if epoch > swa_start_epoch:
            swa_model.update_parameters(model)
        
        model.eval()
        with torch.no_grad():
            for (x_batch, y_batch) in val_loader:
                y_pred = model.forward(x_batch)
                val_loss = criterion(y_pred, y_batch)
                break

        swa_model.eval()
        with torch.no_grad():
            for (x_val, y_val) in val_loader:
                y_pred_val = swa_model.forward(x_val)
                swa_val_loss = criterion(y_pred_val, y_val)
                break

        if (epoch >= reparameterization_start_epoch) and (epoch <= reparameterization_end_epoch):
            if epoch % reallocation_frequency == 0:
                model.reallocate()
                #print(f"Reallocated {model.K.item()} parameters\n")
                run["train/K"].log(model.K.item())
        
        if (val_loss < lowest_val_loss) and (epoch >= reparameterization_end_epoch):
            lowest_val_loss = val_loss
            best_model = copy.deepcopy(model)

        if (swa_val_loss < lowest_swa_val_loss):
            lowest_swa_val_loss = swa_val_loss
            best_swa_model = copy.deepcopy(swa_model)

        run["train/loss"].log(loss)
        run["validation/loss"].log(val_loss)
        if epoch >= swa_start_epoch: 
            run["validation/swa_loss"].log(swa_val_loss)
            run["train/swa_loss"].log(swa_train_loss)
        run["sparsity"].log(model.get_sparsity())
        run["lr"].log(optimizer.param_groups[0]["lr"])

        for l, layer in enumerate(model.layers):
            run[f"train/{l}/sparsity"].log(layer.get_sparsity())

        #if epoch % 10 == 0:
        #    print(f"Epoch {epoch}")
        #    print(f"Training loss: {loss.item()}")
        #    print(f"Validation loss: {val_loss}")
        #    print(f"Sparsity: {model.get_sparsity()} \n")
    
    run.stop()

    if l1 == 0:
        torch.save(model, f"../models/{dataset}/dynamic_sparse_reparameterization/{sparsity}/model_{n+1}.pickle")
        torch.save(best_model, f"../models/{dataset}/dynamic_sparse_reparameterization/{sparsity}/best_model_{n+1}.pickle")
        torch.save(swa_model.state_dict(), f"../models/{dataset}/dynamic_sparse_reparameterization/{sparsity}/swa_model_{n+1}.pickle")
        torch.save(best_swa_model.state_dict(), f"../models/{dataset}/dynamic_sparse_reparameterization/{sparsity}/best_swa_model_{n+1}.pickle")
    else:
        torch.save(model, f"../models/{dataset}/dynamic_sparse_reparameterization/l1/model_{n+1}.pickle")
        torch.save(best_model, f"../models/{dataset}/dynamic_sparse_reparameterization/l1/best_model_{n+1}.pickle")
        torch.save(swa_model.state_dict(), f"../models/{dataset}/dynamic_sparse_reparameterization/l1/swa_model_{n+1}.pickle")
        torch.save(best_swa_model.state_dict(), f"../models/{dataset}/dynamic_sparse_reparameterization/l1/best_swa_model_{n+1}.pickle")
