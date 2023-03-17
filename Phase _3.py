#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:40:59 2023

@author: aisulu
"""

import BA_model_optimised as mod
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import logbin as lg
from collections import Counter
import math
import scipy as sp
from scipy.optimize import curve_fit
import scipy.special as sc
from scipy import stats
from matplotlib.lines import Line2D
from functions import *

plt.style.use('ggplot')
plt.rcParams["font.family"] = "Times New Roman"
red = list(plt.rcParams['axes.prop_cycle'])[0]['color']
blue = list(plt.rcParams['axes.prop_cycle'])[1]['color']
purple = list(plt.rcParams['axes.prop_cycle'])[2]['color']
grey = list(plt.rcParams['axes.prop_cycle'])[3]['color']
yellow = list(plt.rcParams['axes.prop_cycle'])[4]['color']
green = list(plt.rcParams['axes.prop_cycle'])[5]['color']
pink = list(plt.rcParams['axes.prop_cycle'])[6]['color']


#%%
m = 50
r = 30
test = mod.BA_model_opt(m, model = 'EV', r = r)
test.add_nodes(100000)
test_deg = test.degrees()
x, y = getting_prob (test_deg)
x_b, y_b = lg.logbin(test_deg, 1.1)
xs = np.linspace(min(test_deg)+1, max(test_deg), 1000)
y_th = theoretical_dist_EV(xs, m, r, cdf = False, risth = True)
y_th1 = theoretical_dist_EV(xs, m, r, cdf = False, risth = False)
plt.loglog(x, y, '.', label = 'Data', color = blue)
plt.loglog(x_b, y_b, 'x', label = 'Log-binned data', color = 'black')
plt.loglog(xs, y_th, '-',label = 'Theoretical dist.', color = 'orange')
plt.loglog(xs, y_th1, '--',label = 'Theoretical dist. approx.', color = 'red')
plt.xlabel(r'$k$')
plt.ylabel(r'$p(k)$')
plt.legend()



#%%
'''
Plotting the degree distribution for varius m:
    m = 2, 4, 8, 16, 32, 64, 128
with 100000 nodes added to the initial network
'''

ms = [2, 4, 8, 16, 32, 64, 128]
num = 100000

ms0 = mod.BA_model_opt(ms[0], model = 'EV', r = r)
ms1 = mod.BA_model_opt(ms[1], model = 'EV', r = r)
ms2 = mod.BA_model_opt(ms[2], model = 'EV', r = r)
ms3 = mod.BA_model_opt(ms[3], model = 'EV', r = 1)
ms4 = mod.BA_model_opt(ms[4], model = 'EV', r = 1)
ms5 = mod.BA_model_opt(ms[5], model = 'EV', r = 1)
ms6 = mod.BA_model_opt(ms[6], model = 'EV', r = 1)

ms0.add_nodes(num)
ms1.add_nodes(num)
ms2.add_nodes(num)
ms3.add_nodes(num)
ms4.add_nodes(num)
ms5.add_nodes(num)
ms6.add_nodes(num)

data0 = ms0.degrees()
data1 = ms1.degrees()
data2 = ms2.degrees()
data3 = ms3.degrees()
data4 = ms4.degrees()
data5 = ms5.degrees()
data6 = ms6.degrees()

#%%
plot_m_dist(data0, 1.1, num, ms[0], red, model = 'EV', r = r)
plot_m_dist(data1, 1.1, num, ms[1], blue)
plot_m_dist(data2, 1.1, num, ms[2], green)
plot_m_dist(data3, 1.1, num, ms[3], purple)
plot_m_dist(data4, 1.1, num, ms[4], pink)
plot_m_dist(data5, 1.1, num, ms[5], yellow)
plot_m_dist(data6, 1.1, num, ms[6], grey)