#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:55:40 2020

@author: andrine
"""
import pandas as pd

import wave
import numpy as np
import scipy.io.wavfile as wf
import scipy.signal
import pywt
from scipy.signal import resample
import os
MODULE_PATH = os.path.abspath(os.path.join('../..'))
patient_info_path = MODULE_PATH + '/data/Kaggle/external/'
data_path = MODULE_PATH + '/data/Kaggle/processed/'
raw_data_path = MODULE_PATH + '/data/Kaggle/raw/'



from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties



'''
##########################################################
Heping functions extracting the names of the features 
##########################################################
'''



def get_col_names_EEMD_EMD_DWT(feature_type):

    n_levels = 5
    select_imfs = [1,2,3,4,5]

    simple_stats = ['mean' , 'std', 'median', 'min', 'max']
    stats = simple_stats + ['skew']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']

    cols = []
    for n in select_imfs:
        base_1 = f'IMF_{n}_'
        for i in range(n_levels):
            base = base_1 + f'level_{i}_'
            if feature_type == 'simple':
                for stat in simple_stats:
                    cols.append(base + stat)
            elif feature_type == 'HOS':
                for stat in stats_HOS:
                    cols.append(base + stat)
                for s in others:
                    for stat in stats:
                        cols.append(base + s + '_' + stat)
    return cols

def get_col_names_DWT(feature_type):
    decomp_levels = 10
    n_mfcc = 10

    simple_stats = ['mean' , 'std', 'median', 'min', 'max']
    stats = simple_stats + ['skew']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']

    cols = []

    for n in range(decomp_levels):
        base = f'level_{n}_'
        if feature_type == 'simple':
            for stat in simple_stats:
                cols.append(base + stat)

        elif feature_type == 'HOS':
            for stat in stats_HOS:
                cols.append(base + stat)
            for s in others:
                for stat in stats:
                    cols.append(base + s + '_' + stat)

        elif feature_type == 'MFCC':
            mfcc = []
            for i in range(1 , n_mfcc + 1):
                mfcc.append(f'mfcc_{i}')
            for stat in stats_HOS:
                cols.append(base + stat)
            for stat in stats:
                for m in mfcc:
                    cols.append(base + m + '_' + stat)

            for s in others:
                for stat in stats:
                    cols.append(base + s + '_' + stat)
    return cols


def get_col_names_noDecomp(feature_type):
    n_mfcc = 30

    simple_stats = ['mean' , 'std', 'median', 'min', 'max']
    stats = simple_stats + ['skew']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']

    cols = []
    mfcc = []
    if feature_type == 'simple':
        for stat in simple_stats:
            cols.append(stat)

    elif feature_type == 'HOS':
        for stat in stats_HOS:
            cols.append(stat)

        for s in others:
            for stat in stats:
                cols.append(s + '_' + stat)


    elif feature_type == 'MFCC':
        for i in range(1 , n_mfcc + 1):
            mfcc.append(f'mfcc_{i}')
        for stat in stats_HOS:
            cols.append(stat)
        for stat in stats:
            for m in mfcc:
                cols.append(m + '_' + stat)

        for s in others:
            for stat in stats:
                cols.append(s + '_' + stat)
    return cols



def get_col_names_EEMD_EMD(feature_type):
    select_imfs = [1,2,3,4,5]
    n_mfcc = 15

    simple_stats = ['mean' , 'std', 'median', 'min', 'max']
    stats = simple_stats + ['skew']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']

    cols = []
    for n in select_imfs:
        base = f'IMF_{n}_'
        mfcc = []
        for i in range(1 , n_mfcc + 1):
            mfcc.append(f'mfcc_{i}')
        if feature_type == 'simple':
            for stat in simple_stats:
                cols.append(base + stat)
        elif feature_type == 'HOS':
            for stat in stats_HOS:
                cols.append(base + stat)
            for s in others:
                for stat in stats:
                    cols.append(base + s + '_' + stat)
        elif feature_type == 'MFCC':
            for stat in stats_HOS:
                cols.append(base + stat)
            for stat in stats:
                for m in mfcc:
                    cols.append(base + m + '_' + stat)

            for s in others:
                for stat in stats:
                    cols.append(base + s + '_' + stat)

    return cols

'''
##########################################################
Heping functions treating the data
##########################################################
'''


def downsample(data, sr, sr_new = 8000):
    secs = len(data)/sr # Number of seconds in signal X
    new_sr = sr_new
    samps = round(secs*new_sr)     # Number of samples to downsample
    new_data = resample(data, samps)

    return new_data, new_sr


#Will resample all files to the target sample rate and produce a 32bit float array
def read_wav_file(str_filename, target_rate):
    wav = wave.open(str_filename, mode = 'r')
    (sample_rate, data) = extract2FloatArr(wav,str_filename)

    if (sample_rate != target_rate):
        ( _ , data) = resample_2(sample_rate, data, target_rate)

    wav.close()
    return (target_rate, data.astype(np.float32))

def resample_2(current_rate, data, target_rate):
    x_original = np.linspace(0,100,len(data))
    x_resampled = np.linspace(0,100, int(len(data) * (target_rate / current_rate)))
    resampled = np.interp(x_resampled, x_original, data)
    return (target_rate, resampled.astype(np.float32))

# -> (sample_rate, data)
def extract2FloatArr(lp_wave, str_filename):
    (bps, channels) = bitrate_channels(lp_wave)

    if bps in [1,2,4]:
        (rate, data) = wf.read(str_filename)
        divisor_dict = {1:255, 2:32768}
        if bps in [1,2]:
            divisor = divisor_dict[bps]
            data = np.divide(data, float(divisor)) #clamp to [0.0,1.0]
        return (rate, data)

    elif bps == 3:
        #24bpp wave
        return read24bitwave(lp_wave)

    else:
        raise Exception('Unrecognized wave format: {} bytes per sample'.format(bps))

#Note: This function truncates the 24 bit samples to 16 bits of precision
#Reads a wave object returned by the wave.read() method
#Returns the sample rate, as well as the data in the form of a 32 bit float numpy array
#(sample_rate:float, data_data: float[])
def read24bitwave(lp_wave):
    nFrames = lp_wave.getnframes()
    buf = lp_wave.readframes(nFrames)
    reshaped = np.frombuffer(buf, np.int8).reshape(nFrames,-1)
    short_output = np.empty((nFrames, 2), dtype = np.int8)
    short_output[:,:] = reshaped[:, -2:]
    short_output = short_output.view(np.int16)
    return (lp_wave.getframerate(), np.divide(short_output, 32768).reshape(-1))  #return numpy array to save memory via array slicing


def bitrate_channels(lp_wave):
    bps = (lp_wave.getsampwidth() / lp_wave.getnchannels()) #bytes per sample
    return (bps, lp_wave.getnchannels())


def denoise_data(data):
    coeff = pywt.wavedec(data, 'db8')
    sigma = np.std(coeff[-1] )
    n= len( data )
    uthresh = sigma * np.sqrt(2*np.log(n*np.log2(n)))
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='soft' ) for i in coeff[1:] )
    denoised_data =  pywt.waverec( coeff, 'db8' )
    return denoised_data

'''
##########################################################
Heping functions to get data into X, y  pandas format
##########################################################
'''


def remove_unpure_samples(df):
    path = MODULE_PATH + '/data/Kaggle/processed/crackleWheeze/'
    crackle = os.listdir(path + 'crackle/')
    wheeze = os.listdir(path + 'wheeze/')
    none = os.listdir(path + 'none/')
    both = os.listdir(path + 'both/')

    for idx, row in df.iterrows():
        filename = row['name']
        if (filename in wheeze) or (filename in both):
            df = df.drop(idx, axis = 0)

    df.reset_index(drop=True)
    return df



def get_X_y(decomp_type, feature_type = 'all', pure = True,normal = True,
            fs_filter = False,
            fs_auto_encoder = False,
            fs_pca = False, k = 10):
    '''
    Decomp type: noDecomp , EMD, EEMD, DWT, EMD_DWT, EEMD_DWT
    Feature type: simple, HOS or MFCC or all
    k: NB! k has to be 10 or 30 if fs_auto_encoder is True
    '''
    dataset = pd.read_csv(MODULE_PATH + f'/src/features/features/{decomp_type}.csv',  sep=',')
    dataset = dataset.drop('Unnamed: 0', axis = 1)

    if pure:
        dataset = remove_unpure_samples(dataset)
    X, y = dataset.iloc[:, :-2], dataset.iloc[:, -1]


    ##### Only extracting features (simple, HOS or MFCC) #############
    cols = []
    if feature_type == 'all':
        cols = X.columns
    elif decomp_type in ['EMD_DWT', 'EEMD_DWT']:
        if feature_type == 'MFCC':
            print('Action is not valid. EEMD_DWT and EMD_DWT does not have MFCC features')
            cols = X.columns
        else:
            cols = get_col_names_EEMD_EMD_DWT(feature_type)
    elif decomp_type in ['EMD', 'EEMD']:
        cols = get_col_names_EEMD_EMD(feature_type)
    elif decomp_type == 'noDecomp':
        cols = get_col_names_noDecomp(feature_type)
    elif decomp_type == 'DWT':
        cols = get_col_names_DWT(feature_type)


    X, y = dataset[cols] , dataset.iloc[:, -1]

    ##### Normalizing Data #############
    if normal:
        x = X.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        X = pd.DataFrame(x_scaled)

    ##### Performing feature selection #############
    if fs_filter:
        X = SelectKBest(chi2, k=k).fit_transform(X, y)

    if fs_auto_encoder:
        if k == 10:
            cols = [116,  91,  62,  68,  46,  53, 142, 139, 145, 160]
            X, y = X[cols] , dataset.iloc[:, -1]
        elif k == 30:
            cols = [ 27,  79, 210, 136,  82, 203, 184,  87, 141,  60,  95, 103, 198,
            3,  10, 205,  95,  35, 101, 109,   3, 170, 193,  59, 164,  72,
            171,  89, 200,  89]
            X, y = X[cols] , dataset.iloc[:, -1]
        else:
            print('Error: When using Autoencoder for feature selection k has to be either 10 or 30')

    if fs_pca:
        pca = PCA(n_components=k)
        X = pca.fit_transform(X)


    return X, y

'''
##########################################################
Heping functions to plot / extract information, to be put into the report
##########################################################
'''

def plot_conf_matrix(y_test, y_pred, decomp, classifier, dim_red = None):
    colors = ["#F94144", "#F3722C", '#F8961E', '#F9C74F','#90BE6D', '#43AA8B','#577590']
    color_map = {'CNN' : colors[0],
                 'SVM' : colors[1],
                 'kmeans': colors[2],
                 'SOM' : colors[3],
                 'KNN': colors[4],
                 'random_forest':colors[5],
                 'ANN': colors[6]
    }
    
    font = FontProperties(fname = MODULE_PATH + '/src/visualization/CharterRegular.ttf', size = 10, weight = 1000)
    
    colors_2 = ['#FFFFFF', color_map[classifier]]
    cmap_name = 'my colormap'
    font_small = FontProperties(fname =  MODULE_PATH + '/src/visualization/CharterRegular.ttf', size = 6, weight = 1000)

    cm_map = LinearSegmentedColormap.from_list(cmap_name, colors_2)
    class_names = ['crackle', 'no-crackle']
    cm = confusion_matrix(y_test, y_pred)


    f, ax = plt.subplots(1,1) # 1 x 1 array , can also be any other size
    f.set_size_inches(2, 2)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm, annot=True,
                fmt='.2%', cmap=cm_map, xticklabels=class_names,yticklabels=class_names )
    cbar = ax.collections[0].colorbar
    for label in ax.get_yticklabels() :
        label.set_fontproperties(font_small)
    for label in ax.get_xticklabels() :
        label.set_fontproperties(font_small)
    ax.set_ylabel('True Label', fontproperties = font)
    ax.set_xlabel('Predicted Label', fontproperties = font)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)

    for child in ax.get_children():
      if isinstance(child, matplotlib.text.Text):
          child.set_fontproperties(font)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontproperties(font_small)
    if dim_red != None:
        figure_path = MODULE_PATH + '/figures/' 
        f.savefig(figure_path + f'dimRed/{dim_red}_conf_mat.pdf' , bbox_inches='tight')
    else:
        figure_path = MODULE_PATH + '/figures/'
        f.savefig(figure_path + f'{classifier}/{classifier}_conf_matrix_{decomp}.pdf' , bbox_inches='tight')

        
def get_t(y, sr):
    duration = float(len(y)) / sr
    T = 1.0/sr
    N = int(duration / T)
    t = np.linspace(0.0, N*T, N)
    return t

def report_to_latex_table(data):
    avg_split = False
    out = ""
    out += "\\begin{table}\n"
    out += "\\caption{Latex Table from Classification Report}\n"
    out += "\\label{table:classification:report}\n"
    out += "\\centering\n"
    out += "\\begin{tabular}{c | c c c r}\n"
    out += "Class & Precision & Recall & F-score & Support\\\\\n"
    out += "\midrule\n"
    for key, value in data.items():
        if key == 'accuracy':
            continue
        out += key + " & " + str(round(value['precision'] , 3)) + " & " + str(round(value['recall'],3)) + " & " + str(round(value['f1-score'], 3)) + " & " + str(value['support'])
        out += "\\\\\n"
    out += "\\end{tabular}\n"
    out += "\\end{table}"
    print(out)




'''
##########################################################
Heping functions to extract information about the containings of the dataset 
##########################################################
'''
def get_diagnosis_df():
    '''
    Returns pandas dataframe with the PersonID and corresponding diagnose, list of all the types of diseases
    '''
    diagnosis_df = pd.read_csv(patient_info_path + 'patient_diagnosis.csv', sep=",", names=['pId', 'diagnosis'])
    ds = diagnosis_df['diagnosis'].unique()
    return diagnosis_df, ds

def get_filename_info(filename):
    return filename.split('_')

def get_file_info_df():
    file_names = [s.split('.')[0] for s in os.listdir(path = raw_data_path) if '.txt' in s]
    file_paths = [os.path.join(raw_data_path, file_name) for file_name in file_names]

    files_ = []
    for f in file_names:
        df = pd.read_csv(raw_data_path + '/' + f + '.txt', sep='\t', names=['start', 'end', 'crackles', 'wheezes'])
        df['filename'] = f
        #get filename features
        f_features = get_filename_info(f)
        df['pId'] = f_features[0]
        df['ac_mode'] = f_features[3]

        files_.append(df)

    files_df = pd.concat(files_)
    files_df.reset_index()
    files_df.head()
    return files_df

def get_complete_df(target = 'wheeze/crackle'):
    '''
    Returns
    -------
    dataframe with file info, diagnosis of patient

    '''
    diagnosis_df,_ = get_diagnosis_df()
    file_info_df = get_file_info_df()


    file_info_df['pId'] = file_info_df['pId'].astype('int64')
    df = pd.merge(file_info_df, diagnosis_df, on='pId')

    df = df.reset_index()
    df = set_target_of_df(df, target = target)
    i = 0
    new_name = []
    for idx , row in df.iterrows():
        # Idx is the index of the row, and row contains all the data at the given intex
        f = row['filename']

        if idx != 0:
            if df.iloc[idx - 1]['filename'] == f:
                i = i + 1
            else:
                i = 0
        sliced_file_name = f + '_' + str(i) + '.wav'
        new_name.append(sliced_file_name)

    df['filename'] = new_name

    df.drop('level_0', inplace=True, axis=1)
    df.drop('index', inplace=True, axis=1)

    df['len_slice'] = df['end'].sub(df['start'], axis = 0)
    print(df.head(10))

    return df



def set_target_of_df(df, target = 'wheeze/crackle'):
    '''
    Parameters
    ----------
    df : pandas dataframe to be edited.
    target : What is the desired target. Can be set to 'crackle', 'diagnosis', 'wheeze/crackle'
    The default is 'crackle'.

    Returns
    -------
    new dataframe with the specified target.

    '''
    print(target)
    if not(target == 'crackle' or target == 'wheeze/crackle'):
        df = df.reset_index()
        return df

    ab = []
    for idx, row in df.iterrows():
        if (target == 'crackle'):
            if row['crackles'] == 1:
                ab.append('crackle')
            else:
                ab.append('no-crackle')

        elif (target == 'wheeze/crackle'):
            if (row['crackles'] == 1 and row['wheezes'] == 1):
                ab.append('both')
                continue
            elif row['crackles'] == 1:
                ab.append('crackle')
                continue
            elif (row['wheezes'] == 1):
                ab.append('wheeze')
                continue
            else:
                ab.append('none')
                continue

    df['abnormality'] = ab
    df.drop('wheezes', inplace=True, axis=1)
    df.drop('crackles', inplace=True, axis=1)
    df = df.reset_index()
    return df
