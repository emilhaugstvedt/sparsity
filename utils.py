import pickle
import torch
from torch.utils.data import DataLoader
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

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


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors
        
# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height, 
                           facecolor=c, 
                           edgecolor='none'))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch
