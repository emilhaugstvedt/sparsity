import pandas as pd
import numpy as np
from scipy.stats import skew 
import random
import time
import librosa 
import pywt
from utilities import chop_timeseries
import pickle

def get_n_best_levels(data, total_level):
    dwtValues = pywt.wavedec(data, 'db8', level=total_level)
    len_D = len(dwtValues)
    result = []

    for n, c_D in enumerate(dwtValues):
        if n == 0:
            continue
        result.append(c_D)

    return result

def get_t(y, sr):
    n = len(y)
    t = np.linspace(0, 1/ sr, n)
    return t

def get_entropy(data):
    data_nz = data[data != 0]
    return - np.sum(((data_nz**2)*np.log(data_nz**2)))
    
def get_energy(data):  
    N = len(data)
    return np.sum(np.abs(data) ** 2) / N
    
def get_features(filename, data):

    ft1 = librosa.feature.zero_crossing_rate(y=data)[0]
    ft2 = librosa.feature.spectral_rolloff(y=data)[0]
    ft3 = librosa.feature.spectral_centroid(y=data)[0]
    ft4 = librosa.feature.spectral_contrast(y=data)[0]
    ft5 = librosa.feature.spectral_bandwidth(y=data)[0]

    ### Get HOS and simple features 
    ft0_trunc = np.hstack((np.mean(data), np.std(data), skew(data), np.max(data), np.median(data), np.min(data), get_energy(data), get_entropy(data)))
  
    ### Spectral Features 
    ft1_trunc = np.hstack((np.mean(ft1), np.std(ft1), skew(ft1), np.max(ft1), np.median(ft1), np.min(ft1)))
    ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
    ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
    ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
    ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.max(ft5)))
    return pd.Series(np.hstack((ft0_trunc , ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc)))
    

def get_column_names(feature, decomp_levels):
    stats = ['mean' , 'std' , 'skew', 'max', 'median', 'min']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']
    cols = []
    for n in range(decomp_levels):
        base = f'{feature}_level_{n}_'
        for stat in stats_HOS:
            cols.append(base + stat)    
           
        for s in others:
            for stat in stats:
                cols.append(base + s + '_' + stat) 
    return cols


def get_dataset(data, n_levels = 10):
    random.seed(20)

    chopped_timeseries = chop_timeseries(data, 1000)

    print("Timeseries chopped")
    
    df = pd.DataFrame()
    for i, timeserie in enumerate(chopped_timeseries):
        print(f'Processing (DWT): sample number {i} of {len(chopped_timeseries)}')
        
        dwt = get_n_best_levels(timeserie, n_levels)

        row = pd.DataFrame()

        print(f'Processing / feature extract: sample number {i} of {len(dwt)}')
        
        features = pd.DataFrame()
        for (idx, cD) in enumerate(dwt):
            row = pd.DataFrame()
            row['name'] = [i]
            row = row['name'].apply(get_features, data = cD)
            features = pd.concat([features, row], ignore_index=True, axis=1)
        df = pd.concat([df, features], ignore_index=True)

    return df

def main():
    with open('data/Case_2_a_only_basic_DQ', 'rb') as f:
        ((data1_1_df, data1_2_df, data1_3_df),(mean1_df,std1_df)) = pickle.load(f)

    data1_df = pd.concat([data1_1_df,data1_2_df,data1_3_df],axis=0)
    data1_df = data1_df * std1_df + mean1_df

    (imin,_) = next((i, el) for i, el in enumerate(data1_df.HDEP.values) if el < 200)
    data = data1_df.iloc[imin:]

    features = ["ASMPAM1_T", "ASMPAM2_T", "ASMPAM3_T", "FLIAVG", "FLOAVG", "HKLDAV", "ROPA"]

    data = data[features].iloc[1080000:1325000]
    print("Data loaded")
    print("Creating features.")
    start = time.time()
    for feature in features:
        print(f"Creating feature for {feature}")
        df =  get_dataset(data[feature].values, n_levels = 10)
        df.columns = get_column_names(feature, decomp_levels=10)
        df.to_csv(f'features/new_features/{feature}/DWT_complete_samples.csv')
    print(f' Processing finished, total time used = {time.time() - start}')

main()