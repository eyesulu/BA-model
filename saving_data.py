#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:01:20 2023

@author: aisulu
"""

import numpy as np
import BA_model_optimised as mod
'''
This file allows to save degrees for several realisations
A procedure to save the data in npy format is taken from:
    https://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly
The npy format is more efficient when reading the data therefore it is used instead of txt or csv
'''

def save_data(ms, Ns, runs, model, r = 0, multi = False):
    if type(ms) is int:
        ms = [ms]
    if type(Ns) is int:
        Ns = [Ns]
    for m in ms:
        print ('m = %.i' %m)
        for N in Ns:
            print ('N = %.i' %N)
            try:
                degrees_m_N = np.load(f'data/{model}_{m}_{N}.npy')
                degrees_m_N = degrees_m_N.tolist()
            except:
                degrees_m_N = []
            for run in range(runs):
                print (run)
                network = mod.BA_model_opt(m, model = model, r = r, multi = multi)
                network.add_nodes(N)
                degrees_m_N.append(network.degrees())
            np.save(f'data/{model}_{m}_{N}.npy',degrees_m_N)
            
save_data(128,1000000, 100, 'PA')
save_data(256,[100, 1000, 10000, 100000, 1000000], 100, 'PA')
                
