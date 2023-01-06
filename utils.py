import pickle
import torch
from torch.utils.data import DataLoader
import alu_dataset

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

def load_models(path: str, n_models):
    models = []
    for n in range(n_models):
        model = torch.load(f"{path}_{n+1}.pickle")
        models.append(model)
    return models

def load_state_dicts(path: str, n_models: int):
    state_dicts = []
    for n in range(n_models):
        state_dict = torch.load(f"{path}_{n+1}.pickle")
        state_dicts.append(state_dict)
    return state_dicts

def create_models(model: torch.nn.Module, layers, state_dicts: list):
    models = []
    for state_dict in state_dicts:
        model = model(layers)
        model.load_state_dict(state_dict)
        models.append(model)
    return models


def predict(model, x_test, dt, x_mean, x_std, y_mean, y_std):
    prediction = torch.zeros_like(x_test)
    prediction[:, 8:] = x_test[:, 8:]
    prediction[0] = x_test[0]
    for i in range(1,len(x_test)):
        input = (prediction[i-1] - x_mean) / x_std
        with torch.no_grad():
            x_dot = model(input) * y_std + y_mean
        prediction[i, :8] = prediction[i-1, :8] + x_dot * dt
    return prediction

def divergence_detection(models, test_data: alu_dataset.Dataset_alu, test_lengths):
    # Get mean and std from test
    dt = test_data.DT
    x_mean = test_data.x_mean
    x_std = test_data.x_std
    y_mean = test_data.y_mean
    y_std = test_data.y_std

    RFMSE_dict = {length: [] for length in test_lengths}
    divergence_dict = {length: 0 for length in test_lengths}
    for model in models:
        for x_test in test_data.data:
            prediction = predict(model, x_test, dt, x_mean, x_std, y_mean, y_std)
            
            for length in test_lengths:
                norm_error = torch.mean(torch.abs(prediction[:length, :8] - x_test[:length, :8]), axis=0) / x_std[:8]
                norm_error = torch.nan_to_num(norm_error, nan=10, posinf=10, neginf=10)

                if torch.max(norm_error) > 3:
                    divergence_dict[length] += 1
                else:
                    RFMSE_dict[length].append(torch.mean(norm_error))
        
    return RFMSE_dict, divergence_dict

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim