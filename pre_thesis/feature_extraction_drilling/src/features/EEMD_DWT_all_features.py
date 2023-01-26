#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:39:59 2020

@author: andrine
"""

import pandas as pd
import os
import numpy as np
from PyEMD import EEMD  
import pywt
import sys
from scipy.stats import skew 
import random
import time
import librosa 
from scipy.signal import resample 
import pickle
from utilities import chop_timeseries

def get_n_best_IMFs(data, n_imfs, n_sifts, select_imfs):
    m_trials = 2
    eemd = EEMD(trials=m_trials)
    eemd.spline_kind="slinear"
    eemd.FIXE = n_sifts
    # Execute EEMD on S
    result = []
    IMFs = eemd.eemd(data, max_imf = n_imfs)
    for idx in range(len(IMFs)):
        if (idx in select_imfs):
            if idx < len(IMFs):
                result.append(IMFs[idx])
            else:
                result.append(np.zeros(len(data)))
            

    return result

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
    

def get_features(filename , data):
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
    return pd.Series(np.hstack((ft0_trunc, ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc)))


def get_column_names(select_imfs, n_levels):
    stats = ['mean' , 'std' , 'skew', 'max', 'median', 'min']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']
    cols = []
    for n in select_imfs:
        base_1 = f'IMF_{n}_'
        for i in range(n_levels):
            base = base_1 + f'level_{i}_'
            for stat in stats_HOS:
                cols.append(base + stat)    
               
            for s in others:
                for stat in stats:
                    cols.append(base + s + '_' + stat)
               
    #cols.append('name')
    #cols.append('label')    
    return cols


def get_small_dataset(select_imfs, n_samples = 10, n_imfs = 10, n_sifts = 15 , n_levels = 5):
    random.seed(20)
    with open('data/Case_2_a_only_basic_DQ', 'rb') as f:
        ((data1_1_df, data1_2_df, data1_3_df),(mean1_df,std1_df)) = pickle.load(f)

    data1_df = pd.concat([data1_1_df,data1_2_df,data1_3_df],axis=0)
    data1_df = data1_df * std1_df + mean1_df

    (imin,_) = next((i, el) for i, el in enumerate(data1_df.HDEP.values) if el < 200)
    data = data1_df.iloc[imin:]

    print("Data loaded")

    chopped_timeseries = chop_timeseries(data["DHT001_ECD"].values, 1000)

    print("Timeseries chopped")
    
    df = pd.DataFrame()
    for i, timeserie in enumerate(chopped_timeseries):
        print(f'Processing (EEMD): sample number {i} of {len(chopped_timeseries)}')

        IMFs = get_n_best_IMFs(timeserie, n_imfs, n_sifts, select_imfs)

        features = pd.DataFrame()
        print(f'Processing / feature extract: sample number {i} of {n_samples}')
        for (idx_1, imf) in enumerate(IMFs):
            dwt = get_n_best_levels(imf, n_levels)
            for (idx_2, cD) in enumerate(dwt):
                row = pd.DataFrame()
                row['name'] = [i]
                row = row['name'].apply(get_features, data = cD)
            
            features = pd.concat([features, row], axis=1)
            features.reset_index(inplace=True, drop=True)
            
        df = pd.concat([df, features])
   
    
    #names = df.pop(38)
    #labels = df.pop(39)
    #df['name'] = names
    #df['label'] = labelsß
    
    
    #del df['38']
    #del df['39']
    
    df.columns = get_column_names(select_imfs , n_levels)
    
    return df



# If num mfcc = 15 --> label = [121] , filename = [120]
# If num mfcc = 10 --> label = [91], filename = [90]
# If num mfcc = 30 --> label = [211], filename = [210]
def main():
    start = time.time()
    n_samples = 2000 # 12463
    df =  get_small_dataset(select_imfs=[1,2,3,4,5] , n_samples = n_samples, n_imfs = 10, n_sifts = 15)
    print(f'Shape of features: {df.shape}')
    df.to_csv(f'features/new_features/EEMD_DWT_complete_{n_samples}_samples.csv')
    print(f' Processing finished, total time used = {time.time() - start}')

main()