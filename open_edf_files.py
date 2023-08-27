#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificação dados datasets baixados


Created on Mon Jun 26 21:02:23 2023

@author: julia
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% constants / parameters

PATH = '/home/julia/ufrgs/Mestrado/datasets - NI-fECG/abdominal-and-direct-fetal-ecg-database-1.0.0'

#%% filenames

filenames = []

#%% 

file_info = mne.io.read_raw_edf(f'{PATH}/r01.edf')

#%%

channel_names = file_info.ch_names

data = file_info.get_data()

times = file_info.times

#%%

fig, ax = plt.subplots(5, 1, sharex=True)

plt.suptitle('Análise sinais r01.edf')

for i in range(0, 5):
    ax[i].plot(times, data[i, :], label=channel_names[i], color=f'C{i}')

    ax[i].legend(loc='upper right')
    ax[i].grid()
    
    ax[i].set_ylabel(r'$\mu$V')