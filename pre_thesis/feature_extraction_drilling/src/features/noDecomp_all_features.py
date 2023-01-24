import pandas as pd
import os
import numpy as np
import sys
from scipy.stats import skew 
import random
import time
import librosa 
import pickle
from utilities import chop_timeseries

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
    

def get_features(name, data):

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


def get_column_names():
    stats = ['mean' , 'std' , 'skew', 'max', 'median', 'min']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']
    cols = []
    for stat in stats_HOS:
        cols.append(stat)
       
    for s in others:
        for stat in stats:
            cols.append(s + '_' + stat) 
    return cols

def get_small_dataset():

    with open('data/Case_2_a_only_basic_DQ', 'rb') as f:
        ((data1_1_df, data1_2_df, data1_3_df),(mean1_df,std1_df)) = pickle.load(f)

    data1_df = pd.concat([data1_1_df,data1_2_df,data1_3_df],axis=0)
    data1_df = data1_df * std1_df + mean1_df

    (imin,_) = next((i, el) for i, el in enumerate(data1_df.HDEP.values) if el < 200)
    data = data1_df.iloc[imin:]

    print("Data loaded")

    chopped_timeseries = chop_timeseries(data["DHT001_ECD"].iloc[1100000:1300000].values, 10000)

    print("Timeseries chopped")

    df = pd.DataFrame()
    for i, timeserie in enumerate(chopped_timeseries):
        print(f'Processing / feature extract: sample number {i} of {len(chopped_timeseries)}')
  
        row = pd.DataFrame()
        row['name'] = [i]
        row = row['name'].apply(get_features, data=timeserie)

        df = pd.concat([row, df], ignore_index=True)
    
    df.columns = get_column_names()
    
    return df



# If num mfcc = 15 --> label = [121] , filename = [120]
# If num mfcc = 10 --> label = [91], filename = [90]
# If num mfcc = 30 --> label = [211], filename = [210]
# If num mfcc = 40 --> label = [271], filename = [270]
def main():
    start = time.time()
    df =  get_small_dataset()
    print(f'Shape of features: {df.shape}')
    df.to_csv(f'features/new_features/noDecomp_complete.csv', sep=';')
    print(f' Processing finished, total time used = {time.time() - start}')

main()