#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNR

Created on Sat Sep  2 10:23:47 2023

@author: julia
"""

import numpy as np

def calculate_snr(data, time, annotations, interval, qrs_time = 0.04):
    
    ground_truth_time = annotations.onset
    
    max_time = np.max(time)
    
    # append min_time and max_time

    i = 0
    
    snr_data = np.array([])
    
    while i < len(data):
        
        print(i)
    
        this_interval_data = data[i : i + interval]
        this_time_interval = time[i : i + interval]
                
        this_qrs_complex = np.where(
            (ground_truth_time >= np.min(this_time_interval)) &
            (ground_truth_time <= np.max(this_time_interval))
        )[0]
        
        print(this_qrs_complex)

        for j in this_qrs_complex:
            
            qrs_complex_index = np.where(
                (time <= (ground_truth_time[j] + qrs_time / 2)) &
                (time >= (ground_truth_time[j] - qrs_time / 2))
                )[0]
        
            qrs_complex = data[
                    np.min(qrs_complex_index).astype(int) 
                    : 
                    np.max(qrs_complex_index).astype(int)
                    ]
                
            noise = [
                this_interval_data[i]
                for i in range(len(this_interval_data)) 
                if i not in qrs_complex
                ]
        
            print('len', len(qrs_complex))
        
        
            app = np.max(qrs_complex) - np.min(qrs_complex)
            
            std_noise = np.std(noise)
            
            snr_data = np.append(
                snr_data,
                20 * np.log10(app / (4 * std_noise))
                )
            
            
        i += interval
        
        
    return snr_data
        # this_time_interval 
        
    # i = 0
    
    # while i < len(data):
        
    #     partial_time = np.where(time == time[i : i + interval])
        
    #     partial_data = data[i : i + interval]        
        
    #     i += interval
    
    # check if is qrs complex or not in interval of 
    
    
    # calculate app or sigma
    
    # snr = 20 np.log10(app / 4 sigma)
    
    return

