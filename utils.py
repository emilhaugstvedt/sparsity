import pickle
import torch
from torch.utils.data import DataLoader

def load_data(path, batch_size=100) -> DataLoader:
    with open(f"{path}/train.pickle", "rb") as f:
        train_data = pickle.load(f)

    with open(f"{path}/test.pickle", "rb") as f:
        test_data = pickle.load(f)

    with open(f"{path}/val.pickle", "rb") as f:
        val_data = pickle.load(f)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

def rolling_forecast(x_dot_pred, dt):
    pass

def load_models(model, path, n_models):
    models = []
    for n in range(n_models):
        model.load_state_dict(torch.load(f"{path}/model_{n}.pt"))
        models.append(model)
    return models