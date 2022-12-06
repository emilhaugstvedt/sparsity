import torch
from torch.utils.data import dataloader
import sys
import pickle
import os
from tqdm import tqdm
import copy

sys.path.append("../")

import neptune.new as neptune

from plain_net.PlainNet import PlainNet

from utils import load_data

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

### Hyperparameters ###
n_epochs = 100
lr = 1e-3
early_stopping = 100
early_stopping_start_epoch = 30
scheduler_step_size = 10
scheduler_gamma = 0.1

swa_start_epoch = 75

n_models = 10

#### Create model ####
input_size = train_loader.dataset.x.shape[1]
output_size = train_loader.dataset.y.shape[1]

layers = [input_size, 25, 25, 25, 25, output_size]

for n in range(n_models):

    print(f"### Training model {n+1} ###")

    run = neptune.init_run(
        project="emilhaugstvedt/plain-net",
        name=f"model_{n+1}",
        api_token=os.environ["NEPTUNE"]
    )

    run["hyperparameters"] = {
        "n_epochs": n_epochs,
        "lr": lr,
        "layers": layers
    }

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
             0.2 * averaged_model_parameter + 0.8 * model_parameter

    model = PlainNet(layers=layers)
    swa_model = torch.optim.swa_utils.AveragedModel(model)

    lowest_val_loss = 99999
    best_model = model

    lowest_swa_val_loss = 99999
    best_swa_model = swa_model

    # Early stopping counter
    n_early_stopping = 0

    ### Train model ###            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma) # Endre med gamm 0.1 etter 50 epochs
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=1e-5, max_lr=lr*100, step_size_up=20, cycle_momentum=False)
    #swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_strategy="cos", anneal_epochs=10, swa_lr=0.0, 0.001)
    criterion = torch.nn.MSELoss()

    for epoch in tqdm(range(n_epochs)):
        for (x_batch, y_batch) in train_loader:
            model.train()
            optimizer.zero_grad()
            y_pred = model.forward(x_batch)
            swa_train_loss = criterion(swa_model(x_batch), y_batch)
            train_loss = criterion(y_pred, y_batch)
            loss = train_loss
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch > swa_start_epoch:
            swa_model.update_parameters(model)

        model.eval()
        with torch.no_grad():
            for (x_val, y_val) in val_loader:
                y_pred_val = model.forward(x_val)
                val_loss = criterion(y_pred_val, y_val)
                break
        
        swa_model.eval()
        with torch.no_grad():
            for (x_val, y_val) in val_loader:
                y_pred_val = swa_model.forward(x_val)
                swa_val_loss = criterion(y_pred_val, y_val)
                break

        # Log train loss, validation loss and sparsity

        #if (epoch % 10 == 0):
        #        print(f"Epoch {epoch}")
        #        print(f"Train loss: {loss}")
        #        print(f"Validation loss: {val_loss}")
        #        print(f"Sparsity: {model.get_sparsity()} \n")

        run["train/loss"].log(train_loss)
        run["validation/loss"].log(val_loss)
        if epoch >= swa_start_epoch: 
            run["validation/swa_loss"].log(swa_val_loss)
            run["train/swa_loss"].log(swa_train_loss)
        run["sparsity"].log(model.get_sparsity())
        run["lr"].log(optimizer.param_groups[0]["lr"])

        # Early stopping
        if (val_loss < lowest_val_loss):
            lowest_val_loss = val_loss
            best_model= copy.deepcopy(model)
            n_early_stopping = 0
        
        if (val_loss > lowest_val_loss) and (epoch > early_stopping_start_epoch):
            n_early_stopping += 1
        
        if (n_early_stopping > early_stopping):
            print("Early stopping")
            break

        if (swa_val_loss < lowest_swa_val_loss):
            lowest_swa_val_loss = swa_val_loss
            best_swa_model = copy.deepcopy(swa_model)

    run.stop()

    #torch.save(model, f"../models/{dataset}/plain_net/model_{n+1}.pickle")
    #torch.save(best_model, f"../models/{dataset}/plain_net/best_model_{n+1}.pickle")
    torch.save(swa_model.state_dict(), f"../models/{dataset}/plain_net/swa_model_{n+1}.pickle")
    torch.save(best_swa_model.state_dict(), f"../models/{dataset}/plain_net/best_swa_model_{n+1}.pickle")