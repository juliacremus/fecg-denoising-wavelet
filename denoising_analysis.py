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

STEP = 512  # subdived data in STEP parts

#%% Get data 

filenames = glob.glob(f'{PATH}*.edf')

# get data info class from mne

file_info = mne.io.read_raw_edf(f'{PATH}/r01.edf')
annotations = mne.read_annotations(f'{PATH}/r01.edf')

channel_names = file_info.ch_names

data = file_info.get_data()

times = file_info.times

N = len(data[0])

#%%

snr_data = calculate_snr(data[0], times, annotations, STEP, qrs_time = 0.04)


#%% Transform data with wavelets and filter 

i = 0

transformed_data = np.array([])

while i < N:

    # Transform data with wavelet    
    haar = HaarTransform(data[0][i:i + STEP], levels, f)
    details = haar.run_cascade_multiresolution_transform()
    
   
    # Filter coefs
    # 'minimax', 'alice', 'han', 'universal'
    data_filtered = filtering_coef(N, 
                                details, 
                                levels, 
                                'alice', 
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

ax.plot(transformed_data)