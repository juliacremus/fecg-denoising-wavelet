#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detects QRS and return array of peaks by time

Created on Sat Sep  2 10:23:47 2023

@author: julia
"""


def get_qrs_complex(fs, data):
    
    detectors = Detectors(fs)
    
    #%%
    
    r_peaks = detectors.christov_detector(data[0])
    
    peaks_in_data = []
    
    
    for i in r_peaks:
        
        peaks_in_data.append(data[0][i])
    
    
    return 