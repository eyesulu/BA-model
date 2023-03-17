#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:34:28 2023

@author: aisulu
"""

import numpy as np
import matplotlib.pyplot as plt
import logbin as lg

def get_logbin(data, scale, zeros = False):
    max_num = 0
    ys = []
    for run in data:
        x, y = lg.logbin(run, scale = scale, zeros = True)
        ys.append(y)
        if len(x) > max_num:
            max_num = len(x)
            xs = x
    for i in range (len(data)):
        while len(ys[i]) < max_num:
            ys[i] = np.append(ys[i], 0)
    mean_ys = np.mean(ys, axis = 0)
    std_ys = np.std(ys, axis = 0)
    range_ys = np.subtract(np.amax(ys, axis = 0) , np.amin(ys, axis = 0))
    return xs, mean_ys, std_ys, range_ys

def get_data(m, N, model, scale, cdf = False):
    degrees = np.load(f'data/{model}_{m}_{N}.npy')
    if cdf == False:
        ks, prob, std_prob, range_prob = get_logbin(degrees, scale)
        plt.loglog(ks, prob, '.')
        plt.errorbar(ks, prob, yerr = std_prob, fmt='.', capsize = 2, 
                     label = 'm = %.i, N = %.i' %(m, N) )
        
get_data(2, 1000000, 'PA', 1.1)
get_data(8, 1000000, 'PA', 1.1)
get_data(32, 1000000, 'PA', 1.1)
        
    
    