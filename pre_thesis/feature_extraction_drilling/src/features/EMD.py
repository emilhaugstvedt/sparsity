import PyEMD as EMD
import numpy as np
from scipy.stats import skew
import pandas as pd

def get_EMD(data, n_imfs=None):
    emd = EMD.EMD()
    imfs = emd(data)
    if n_imfs is None:
        return imfs
    return imfs[:n_imfs]

def get_entropy(data):
    return - np.sum(((data**2)*np.log(data**2)))
    
def get_energy(data):  
    N = len(data)
    return np.sum(np.abs(data) ** 2) / N

def get_features(data, n_imfs=None):
    features = [np.mean(data), np.std(data), skew(data), np.max(data), np.median(data), np.min(data), get_energy(data), get_entropy(data)]
    return pd.Series(features)

def get_column_names(select_imfs):
    column_names = []
    for n in select_imfs:
        base = f'IMF_{n}'
        for feature in ['mean', 'std', 'skew', 'max', 'median', 'min', 'energy', 'entropy']:
            column_names.append(base + '_' + feature)
    return column_names
    
def get_EMD_features(data, n_imfs):
    emd = EMD.EMD()
    features = pd.DataFrame()
    for d in data:
        imfs = emd(d)[:n_imfs]
        imf_features = pd.DataFrame()
        for imf in imfs:
            imf_features = pd.concat([imf_features, get_features(imf)], axis=0, ignore_index=True)
        features = pd.concat([features.T, imf_features], axis=1, ignore_index=True).T

    features.columns = get_column_names(range(len(imfs)))
    return features