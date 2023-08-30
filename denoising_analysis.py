#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Denoising analysis

Created on Wed Aug 16 10:55:02 2023

@author: julia
"""

import mne
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from haar_transform.haar_tranform import HaarTransform

#%% support functions


def tresholding(N, details, level, type_threshold, type_thresholding):
    
    if type_threshold == 'minimax':
        threshold = minimax_threshold(N, details)
    elif type_threshold == 'universal':
        threshold = universal_threshold(N, details)
    elif type_threshold == 'han':
        threshold = han_etal_threshold(N, details, level)
    elif type_threshold == 'alice':
        threshold = spc_shrink(details)
    
    if type_thresholding == 'hard':
        return_value = details
        
    elif type_thresholding == 'soft':
        return_value = (details / np.abs(details)) * (
            np.abs(details) - threshold
        )
        
    details = np.where(
        np.abs(details) >= threshold, 
        return_value, 
        0
    )
    
    return details


def minimax_threshold(N, details):
    
    threshold = sigma(details) * (0.3936 + 0.1829 * np.log2(N))
    
    return threshold

def universal_threshold(N, details):
    
    threshold = sigma(details) * np.sqrt(2 * np.log(N))
    
    return threshold

def han_etal_threshold(N, details, level, L=10):
    
    if level == 1:
        threshold = sigma(details) * np.sqrt(2 * np.log(N))
    elif level > 1 and level < L:
        threshold = sigma(details) * np.sqrt(2 * np.log(N)) / np.log(level + 1)
    elif level == L:
        threshold = sigma(details) * np.sqrt(2 * np.log(N)) / np.sqrt(level)
    
    return threshold

def spc_shrink(details, coef_d = 2):
    
    # padrao alpha = 1%
    
    # alice treshholding
    
    data = np.copy(details)
    
    keep_calculating = True
    
    N = len(data)
    mean_wavelet = np.mean(data)

    while keep_calculating:
        
        deviation_s = np.sqrt(
                (1 / (N - 1)) * np.sum(np.square(data - mean_wavelet))
            )
        
        LCL = - coef_d * deviation_s
        UCL = coef_d * deviation_s
    
        data = data[(data >= LCL) & (data <= UCL)]
    
        if len(data) == N:
            keep_calculating = False
        else:
            N = len(data)
    
    return coef_d * deviation_s

def sigma(details):
    
    mean_detail = np.median(details)
    
    sigma = 1.4826 * np.median(details - mean_detail)
    
    return sigma

#%%

def SNR_article (signal, noise):
    
    deviation_signal = np.std(noise)
    
    amplitude = signal  # have to get aplitude ofpeak to peak
    
    #     For the sake of the SNR calcula-
    # tion only, the estimation of the standard deviation of the noise was
    # conceived as the median of the standard deviations computed on
    # each interval between two foetal beats. This approach considers
    # the presence of both the noise and potential residual mECG (the
    # latter only in the real dataset).
    
    # Moreover, before the WD, we computed the Appf of a given sig-
    # nal on its average QRS complex, obtained by synchronized averag-
    # ing, to reduce the inter-beat variability. 
    
    snr_values = 20 * np.log(amplitude /  4 * deviation_signal)
    
    return snr_values
#%% Contants / parameters

PATH = '/home/julia/Documents/temp_julia/denoising_analysis/data'

f = 1 / 2

levels = 7

#%% Get data 

# filenames (in future make a loop in filenames)

filenames = glob.glob(f'{PATH}*.edf')

# get data info class from mne

file_info = mne.io.read_raw_edf(f'{PATH}/r01.edf')

# get specific data 

channel_names = file_info.ch_names

data = file_info.get_data()

times = file_info.times

#%% Plot raw data 

fig, ax = plt.subplots()

ax.plot(data[0], label='Direct')

ax.grid()

ax.legend()

#%%   get parameters from data

N = len(data[0])


# escolher N' como potencia de 2, 

# escolher a epoca como um periodo entre um pico e outro ECG
# caso nao tenha no artigo do wavelet
#%%


intervalo = 512

#%% Transform data with wavelets

haar = HaarTransform(data[0][0:intervalo], levels, f)

details = haar.run_cascade_multiresolution_transform()

#%% Plot raw data 

fig, ax = plt.subplots(2)

ax[0].plot(data[0][:intervalo], label='Direct')

# ax.plot(times[:intervalo], data[0][:intervalo], label='Direct')

ax[1].plot(details[:int(intervalo/2)])

ax[0].grid()

ax[0].legend()

#%%

data_filtered = tresholding(N, details, levels, 'alice', 'hard')

#%%

reconstruct_data = HaarTransform(data_filtered, levels, f)

filtered_signal = reconstruct_data.run_cascade_multiresolution_inv_transform()
#%% Plot raw data 

fig, ax = plt.subplots(3)

ax[0].plot(data[0][:intervalo], label='Direct')

# ax.plot(times[:intervalo], data[0][:intervalo], label='Direct')

ax[1].plot(details[:int(intervalo/2)])

ax[2].plot(filtered_signal)

ax[0].grid()

ax[0].legend()

#%%

peaks, _ = find_peaks(np.abs(filtered_signal), height=0)

print()

#%%

fig, ax = plt.subplots()

ax.plot(peaks, filtered_signal[peaks], marker = 'x')

ax.plot(filtered_signal)

#%%

norm_filtered_signal = filtered_signal / np.max(filtered_signal)

linear_coef = np.array([0])

for index in range(len(peaks) - 1):
    
    this_value = norm_filtered_signal[index]
    next_value = norm_filtered_signal[index + 1]
    
    x = np.array([peaks[index], peaks[index + 1]])
    y = np.array([this_value, next_value])
    
    A = np.vstack([x, np.ones(len(x))]).T

    a, b = np.linalg.lstsq(A, y)[0]
    
    linear_coef = np.append(linear_coef, np.array([a]))
    
#%%

fig, ax = plt.subplots()

ax.plot(peaks, linear_coef)    

ax.plot(peaks, filtered_signal[peaks], marker = 'x')    

#%%
# In the real dataset, where
# mECG residuals could be present after the fECG extraction algo-
# rithm, the QRS averaging involved only those complexes exhibiting
# a Pearson’s correlation coeﬃcient above a given threshold, empiri-
# cally chosen to be 0.6. If the number of correlated beats was lower
# than four, the signal was treated as non-deterministic, then substi-
# tuting Appf by the computation of four times the median standard
# deviation of such beats.

#%% get QRS complex in the wavelet denoising way

# get peaks

# in the interval of peak  +40 ms is considered to be QRS complex
# outside of it calculates sigma of noise
# if in complex calculates amplitude
# and then calculates SNR

# for accuracy calculation                      we have to get this code
# https://iopscience.iop.org/article/10.1088/0967-3334/35/8/1569/pdf
# but its on matlab  grrr

#%%

def get_peaks(data, PEARSON_THRESHOLD = 0.6):
    
    # Get peaks
    
    # Calculate linear correlation of peaks (lower and up)
    
    # If correlation is bigger than threshold than is consider a qrs complex
    
    peaks, _ = find_peaks(data, height=0)
    
    print()
    
    return peaks


def calculate_snr():
    
    
    # If interval is qrs complex calculates App
    
    # Else calculates sigma of noise
    
    

    # maybe we have to calculate the median value of SNR
    # in the whole time data
    # and plot comparisson between mean values with diffente thheshold methossss
    
    return 
#%% #%% Filter data method 2


#%% Plot data 