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

from haar_transform.haar_tranform import HaarTransform

#%% support functions

def spc_shrink():
    
    # alice treshholding
    
    return

def hard_tresholding(N, details, level, type_threshold, type_thresholding):
    
    if type_threshold == 'minimax':
        threshold = minimax_threshold(N, details)
    elif type_threshold == 'universal':
        threshold = universal_threshold(N, details)
    elif type_threshold == 'universal':
        threshold = han_etal_threshold(N, details, level)
    
    
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

def han_etal_threshold(N, details, level, L):
    
    if level == 1:
        threshold = sigma(details) * np.sqrt(2 * np.log(N))
    elif level > 1 and level < L:
        threshold = sigma(details) * np.sqrt(2 * np.log(N)) / np.log(level + 1)
    elif level == L:
        threshold = sigma(details) * np.sqrt(2 * np.log(N)) / np.sqrt(level)
    
    return threshold

def sigma(details):
    
    mean_detail = np.median(details)
    
    sigma = 1.4826 * np.median(details - mean_detail)
    
    return sigma
# acho que vou fazer a filtragem com uma
# classe que possui o vetor de haaar transformado
# e aí tem metodos tipo 
#%% Contants / parameters

PATH = '/home/julia/Documents/temp_julia/denoising_analysis/data'

f = 1 / 2

levels = 3

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


intervalo = 1024

#%% Transform data with wavelets

haar = HaarTransform(data[0][0:intervalo], levels, f)

foward_transform_array = haar.run_foward_transform()

#%% Plot raw data 

fig, ax = plt.subplots()

ax.plot(times, data[0], label='Direct')

# ax.plot(times[:intervalo], data[0][:intervalo], label='Direct')

ax.plot(times[0:intervalo][::2], foward_transform_array[:int(intervalo/2)])

ax.grid()

ax.legend()

#%% Filter data method 1

#%% #%% Filter data method 2


#%% Plot data 