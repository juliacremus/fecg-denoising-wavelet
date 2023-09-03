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

from ecgdetectors import Detectors

from support.threshold import filtering_coef
from support.qrs_complex import get_qrs_complex
from support.snr_metric import calculate_snr
from haar_transform.haar_tranform import HaarTransform

#%% Contants / parameters

PATH = '/home/julia/Documentos/ufrgs/Mestrado/fECG - research/scripts/fecg-denoising/data'

f = 1 / 2  # 

levels = 7

fs = 250

STEP = 1024  # subdived data in STEP parts

#%% Get data 

filenames = glob.glob(f'{PATH}*.edf')

# get data info class from mne

file_info = mne.io.read_raw_edf(f'{PATH}/r04.edf')
annotations = mne.read_annotations(f'{PATH}/r04.edf')

channel_names = file_info.ch_names

data = file_info.get_data()

direct_fecg = data[0]

times = file_info.times

N = len(direct_fecg)



# #%%

# snr_data = calculate_snr(direct_fecg, times, annotations, STEP, qrs_time = 0.04)

# print('mean snr', np.mean(snr_data))
# print('std snr', np.std(snr_data))
#%% Transform data with wavelets and filter 

i = 0

transformed_data = np.array([])

while i < N:

    # Transform data with wavelet    
    haar = HaarTransform(direct_fecg[i:i + STEP], levels, f)
    details = haar.run_cascade_multiresolution_transform()
    
   
    # Filter coefs
    # 'minimax', 'alice', 'han', 'universal'
    data_filtered = filtering_coef(N, 
                                details, 
                                levels, 
                                'han', 
                                'hard'
                                )
    
    # Recosntruct signal filtered
    reconstruct_data = HaarTransform(data_filtered, levels, f)
    filtered_signal = reconstruct_data.run_cascade_multiresolution_inv_transform()
    
    # Append in array
    transformed_data = np.append(transformed_data, filtered_signal)
    
    i += STEP

#%%

fig, ax = plt.subplots()

ax.plot(times, direct_fecg, c = 'gray')

ax.plot(times[:len(transformed_data)], transformed_data, c= 'black')

#%%

fig, ax = plt.subplots()

ax.plot(direct_fecg, c = 'gray')

ax.plot(transformed_data, c= 'black')

#%%

# 23.004549379643827
# mean snr 22.19860321995984
#%%

snr_alice = calculate_snr(
    direct_fecg[:len(transformed_data)], transformed_data
)

print('mean snr', snr_alice)
# print('std snr', np.std(snr_alice))

# #%%

# fig, ax = plt.subplots()

# ax.plot(snr_alice)

# ax.plot(snr_data, c='b')

# #%%

# detectors = Detectors(STEP)

# #%%

# r_peaks = detectors.christov_detector(transformed_data)

# peaks = [transformed_data[i] for i in r_peaks]
# peaks_time = [times[i] for i in r_peaks]

# data_p = [direct_fecg[np.where(times == i)]  for i in annotations.onset]

# #%%

# fig, ax = plt.subplots()


# ax.plot(peaks_time, peaks, marker = '+', c = 'red')

# ax.plot(annotations.onset, data_p, marker = 'o', c = 'blue')

#%%
# peaks_in_data = []


# for i in r_peaks:
    
#     peaks_in_data.append([i])