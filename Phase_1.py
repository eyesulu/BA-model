#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:10:17 2023

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
m = 2
r = 1
test = mod.BA_model_opt(m, model = 'PA', multi = True)
test1 = mod.BA_model_opt(m, model = 'PA', multi = False)
test.add_nodes(100000)
test1.add_nodes(100000)
test_deg = test.degrees()
test1_deg = test1.degrees()
x, y = getting_prob (test_deg)
x1, y1 = getting_prob (test1_deg)
x_b, y_b = lg.logbin(test_deg, 1.1)
x_b1, y_b1 = lg.logbin(test1_deg, 1.1)
xs = np.linspace(min(test_deg), max(test_deg), 1000)
y_th = theoretical_dist_PA(xs, m, cdf = False)
plt.loglog(x, y, '.', label = 'Data', color = blue)
plt.loglog(x_b, y_b, 'x', label = 'Log-binned data', color = 'black')
plt.loglog(x_b1, y_b1, 'x', label = 'Log-binned data', color = 'green')

plt.loglog(xs, y_th, '--',label = 'Theoretical dist.', color = 'orange')
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

ms0 = mod.BA_model_opt(ms[0],model = 'PA',multi = False)
ms1 = mod.BA_model_opt(ms[1],model = 'PA',multi = False)
ms2 = mod.BA_model_opt(ms[2],model = 'PA',multi = False)
ms3 = mod.BA_model_opt(ms[3],model = 'PA',multi = False)
ms4 = mod.BA_model_opt(ms[4],model = 'PA',multi = False)
ms5 = mod.BA_model_opt(ms[5],model = 'PA',multi = False)
ms6 = mod.BA_model_opt(ms[6],model = 'PA',multi = False)

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

plot_m_dist(data0, 1.3, num, ms[0], red)
plot_m_dist(data1, 1.3, num, ms[1], blue)
plot_m_dist(data2, 1.3, num, ms[2], green)
plot_m_dist(data3, 1.3, num, ms[3], purple)
plot_m_dist(data4, 1.3, num, ms[4], pink)
plot_m_dist(data5, 1.3, num, ms[5], yellow)
plot_m_dist(data6, 1.3, num, ms[6], grey)
#%%
plot_m_dist(data0, 1.3, num, ms[0], red, cdf = True)
plot_m_dist(data1, 1.3, num, ms[1], blue, cdf = True)
plot_m_dist(data2, 1.3, num, ms[2], green, cdf = True)
plot_m_dist(data3, 1.3, num, ms[3], purple, cdf = True)
plot_m_dist(data4, 1.3, num, ms[4], pink, cdf = True)
plot_m_dist(data5, 1.3, num, ms[5], yellow, cdf = True)
plot_m_dist(data6, 1.3, num, ms[6], grey, cdf = True)
#%%
'''
Plotting the degree distribution with m = 64 and 
varius number of added nodes, N:
    N = 1000, 5000, 10000, 15000, 20000, 50000, 100000
'''
m = 3
nums = [1000, 5000, 10000, 15000, 20000, 50000, 100000]

n0 = mod.BA_model_opt(m)
n1 = mod.BA_model_opt(m)
n2 = mod.BA_model_opt(m)
n3 = mod.BA_model_opt(m)
n4 = mod.BA_model_opt(m)
n5 = mod.BA_model_opt(m)
n6 = mod.BA_model_opt(m)

n0.add_nodes(nums[0])
n1.add_nodes(nums[1])
n2.add_nodes(nums[2])
n3.add_nodes(nums[3])
n4.add_nodes(nums[4])
n5.add_nodes(nums[5])
n6.add_nodes(nums[6])

datan0 = n0.degrees()
datan1 = n1.degrees()
datan2 = n2.degrees()
datan3 = n3.degrees()
datan4 = n4.degrees()
datan5 = n5.degrees()
datan6 = n6.degrees()
#%%
plot_N_dist(datan0, 1.2, nums[0], m, red)
plot_N_dist(datan1, 1.2, nums[1], m, blue)
plot_N_dist(datan2, 1.2, nums[2], m, green)
plot_N_dist(datan3, 1.2, nums[3], m, purple)
plot_N_dist(datan4, 1.2, nums[4], m, pink)
plot_N_dist(datan5, 1.2, nums[5], m, yellow)
plot_N_dist(datan6, 1.2, nums[6], m, grey, plot_th = True)
#%%
plot_N_dist(datan0, 1.2, nums[0], m, red, cdf = True)
plot_N_dist(datan1, 1.2, nums[1], m, blue, cdf = True)
plot_N_dist(datan2, 1.2, nums[2], m, green, cdf = True)
plot_N_dist(datan3, 1.2, nums[3], m, purple, cdf = True)
plot_N_dist(datan4, 1.2, nums[4], m, pink, cdf = True)
plot_N_dist(datan5, 1.2, nums[5], m, yellow, cdf = True)
plot_N_dist(datan6, 1.2, nums[6], m, grey, plot_th = True, cdf = True)

#%%
'''
Doing a statistical test on the theoretical distribution 
and the numerical distribution of degrees for varius m:
    m = m = 2, 4, 8, 16, 32, 64, 128
with 100000 nodes added to the initial network
'''
print ('Testing the dist for different m: ')
stat_test(data0, ms[0], num, 1.3, 0.05, short_print = True)
stat_test(data1, ms[1], num, 1.3, 0.05, short_print = True)
stat_test(data2, ms[2], num, 1.3, 0.05, short_print = True)
stat_test(data3, ms[3], num, 1.3, 0.05, short_print = True)
stat_test(data4, ms[4], num, 1.3, 0.05, short_print = True)
stat_test(data5, ms[5], num, 1.3, 0.05, short_print = True)
stat_test(data6, ms[6], num, 1.3, 0.05, short_print = True)
print ()
print ('Testing the dist for different N: ')
stat_test(datan0, m, nums[0], 1.3, 0.05, short_print = True)
stat_test(datan1, m, nums[1], 1.3, 0.05, short_print = True)
stat_test(datan2, m, nums[2], 1.3, 0.05, short_print = True)
stat_test(datan3, m, nums[3], 1.3, 0.05, short_print = True)
stat_test(datan4, m, nums[4], 1.3, 0.05, short_print = True)
stat_test(datan5, m, nums[5], 1.3, 0.05, short_print = True)
stat_test(datan6, m, nums[6], 1.3, 0.05, short_print = True)
'''
All p-values are >= 0.01 hence they all follow the theoretical distribution
'''
#%%
print ('Testing the dist for different m: ')
stat_test(data0, ms[0], num, 1.3, 0.05, cdf = True, short_print = True)
stat_test(data1, ms[1], num, 1.3, 0.05, cdf = True, short_print = True)
stat_test(data2, ms[2], num, 1.3, 0.05, cdf = True, short_print = True)
stat_test(data3, ms[3], num, 1.3, 0.05, cdf = True, short_print = True)
stat_test(data4, ms[4], num, 1.3, 0.05, cdf = True, short_print = True)
stat_test(data5, ms[5], num, 1.3, 0.05, cdf = True, short_print = True)
stat_test(data6, ms[6], num, 1.3, 0.05, cdf = True, short_print = True)
print ()
print ('Testing the dist for different N: ')
stat_test(datan0, m, nums[0], 1.3, 0.05, cdf = True, short_print = True)
stat_test(datan1, m, nums[1], 1.3, 0.05, cdf = True, short_print = True)
stat_test(datan2, m, nums[2], 1.3, 0.05, cdf = True, short_print = True)
stat_test(datan3, m, nums[3], 1.3, 0.05, cdf = True, short_print = True)
stat_test(datan4, m, nums[4], 1.3, 0.05, cdf = True, short_print = True)
stat_test(datan5, m, nums[5], 1.3, 0.05, cdf = True, short_print = True)
stat_test(datan6, m, nums[6], 1.3, 0.05, cdf = True, short_print = True)
#%%
'''
Investigating the finite-size effect
'''

k_max_num0 = []
k_max_err0 = []
for k in range(len(nums)):
    k_max_vals = []
    for i in range(10):
        a = mod.BA_model_opt(ms[0])
        a.add_nodes(nums[k])
        k_max_vals.append(np.max(a.degrees()))
    k_max_num0.append(np.average(k_max_vals))
    k_max_err0.append(np.max(k_max_vals) - np.min(k_max_vals))

k_max_num1 = []
k_max_err1 = []
for k in range(len(nums)):
    k_max_vals = []
    for i in range(10):
        a = mod.BA_model_opt(ms[5])
        a.add_nodes(nums[k])
        k_max_vals.append(np.max(a.degrees()))
    k_max_num1.append(np.average(k_max_vals))
    k_max_err1.append(np.max(k_max_vals) - np.min(k_max_vals))

k_max_num2 = []
k_max_err2 = []
for k in range(len(nums)):
    k_max_vals = []
    for i in range(10):
        a = mod.BA_model_opt(ms[6])
        a.add_nodes(nums[k])
        k_max_vals.append(np.max(a.degrees()))
    k_max_num2.append(np.average(k_max_vals))
    k_max_err2.append(np.max(k_max_vals) - np.min(k_max_vals))

k_max_num3 = []
k_max_err3 = []
for k in range(len(nums)):
    k_max_vals = []
    for i in range(10):
        a = mod.BA_model_opt(ms[2])
        a.add_nodes(nums[k])
        k_max_vals.append(np.max(a.degrees()))
    k_max_num3.append(np.average(k_max_vals))
    k_max_err3.append(np.max(k_max_vals) - np.min(k_max_vals))

#%%
numx = np.linspace(min(nums), max(nums), 1000)

fig, axes = plt.subplots(figsize = [7, 5])
    
axes.loglog(numx, exp_kmax(numx, ms[0]),'-', color = blue, label = r'Theoretical $k_{max}$ for $m = %.i$' %ms[0])
axes.errorbar(nums, k_max_num0, yerr = k_max_err0, fmt = '.',color = blue, label = r'Numerical $k_{max}$ for $m = %.i$' %ms[0],capsize = 2)
po0, po_cov0 = sp.optimize.curve_fit(power_func,nums, k_max_num0)
axes.loglog(numx, power_func(numx, po0[0], po0[1]),'--', label = 'Fit function for $m = %.i$' %ms[0],linewidth = 0.8, color = blue )
axes.text(1200,k_max_num0[0]-20, r'$\propto N^{%.3f}$' % po0[1], color = blue)

axes.loglog(numx, exp_kmax(numx, ms[5]),'-', color = red, label = r'Theoretical $k_{max}$ for $m = %.i$' %ms[5])
axes.errorbar(nums, k_max_num1, yerr = k_max_err1, fmt = '.',color = red, label = r'Numerical $k_{max}$ for $m = %.i$' %ms[5],capsize = 2)
po1, po_cov1 = sp.optimize.curve_fit(power_func,nums, k_max_num1)
axes.loglog(numx, power_func(numx, po1[0], po1[1]),'--', label = 'Fit function for $m = %.i$' %ms[5],linewidth = 0.8, color = red )
axes.text(1200,k_max_num1[0]-60, r'$\propto N^{%.3f}$' % po1[1], color = red)

axes.loglog(numx, exp_kmax(numx, ms[6]),'-', color = green, label = r'Theoretical $k_{max}$ for $m = %.i$' %ms[6])
axes.errorbar(nums, k_max_num2, yerr = k_max_err2, fmt = '.',color = green, label = r'Numerical $k_{max}$ for $m = %.i$' %ms[6], capsize = 2)
po2, po_cov2 = sp.optimize.curve_fit(power_func,nums, k_max_num2)
axes.loglog(numx, power_func(numx, po2[0], po2[1]),'--', label = 'Fit function for $m = %.i$' %ms[6],linewidth = 0.8, color = green )
axes.text(1200,k_max_num2[0]-60, r'$\propto N^{%.3f}$' % po2[1], color = green)

axes.loglog(numx, exp_kmax(numx, ms[2]),'-', color = purple, label = r'Theoretical $k_{max}$ for $m = %.i$' %ms[2])
axes.errorbar(nums, k_max_num3, yerr = k_max_err3, fmt = '.',color = purple, label = r'Numerical $k_{max}$ for $m = %.i$' %ms[2], capsize = 2)
po3, po_cov3 = sp.optimize.curve_fit(power_func,nums, k_max_num3)
axes.loglog(numx, power_func(numx, po3[0], po3[1]),'--', label = 'Fit function for $m = %.i$' %ms[6],linewidth = 0.8, color = purple )
axes.text(1200,k_max_num3[0]-10, r'$\propto N^{%.3f}$' % po3[1], color = purple)

legend_elements1 = [Line2D([0], [0],linestyle = '-', color='black', label=r'Theoretical $k_{max}$'),
                   Line2D([0], [0],linestyle ='--', color = 'black', label=r'Fit function for $k_{max}$'),
                   Line2D([0], [0], marker = '.',linestyle = 'None', label = r'Measured $k_{max}$', color = 'black')]
legend_elements2 = [Line2D([0], [0], color=blue, lw=4, label = r'$m = %.i$' %ms[0]), 
                    Line2D([0], [0], color=purple, lw=4, label = r'$m = %.i$' %ms[2]),
                    Line2D([0], [0], color=red, lw=4, label = r'$m = %.i$' %ms[5]),
                    Line2D([0], [0], color=green, lw=4, label = r'$m = %.i$' %ms[6])]
legend1 = plt.legend(handles = legend_elements1, loc='upper left')
legend2 = plt.legend(handles = legend_elements2,  loc='lower right')
axes.add_artist(legend1)
axes.add_artist(legend2)

axes.set_xlabel(r'$N$')
axes.set_ylabel(r'$k_{max}$')


fig.tight_layout()
#%%
'''
Collapsing the data
'''


datan0 = n0.degrees()
datan1 = n1.degrees()
datan2 = n2.degrees()
datan3 = n3.degrees()
datan4 = n4.degrees()
datan5 = n5.degrees()
datan6 = n6.degrees()
#%%
collapse(data0, num, ms[0], red,scale = 1.3, cdf = False)
collapse(data1, num, ms[1], blue,scale = 1.3, cdf = False)
collapse(data2, num, ms[2], green,scale = 1.3, cdf = False)
collapse(data3, num, ms[3], purple,scale = 1.3, cdf = False)
collapse(data4, num, ms[4], pink,scale = 1.3, cdf = False)
collapse(data5, num, ms[5], yellow,scale = 1.3, cdf = False)
collapse(data6, num, ms[6], grey,scale = 1.3, cdf = False)

collapse(datan0, nums[0], m, 'red',scale = 1.3, cdf = False)
collapse(datan1, nums[1], m, 'blue',scale = 1.3, cdf = False)
collapse(datan2, nums[2], m, 'green',scale = 1.3, cdf = False)
collapse(datan3, nums[3], m,'purple',scale = 1.3, cdf = False)
collapse(datan4, nums[4], m, 'pink',scale = 1.3, cdf = False)
collapse(datan5, nums[5], m, 'yellow',scale = 1.3, cdf = False)
collapse(datan6, nums[6], m, 'grey',scale = 1.3, cdf = False)

#plt.legend()
#%%
collapse(data0, num, ms[0], red,scale = 1.3, cdf = False,kmaxnum = False)
collapse(data1, num, ms[1], blue,scale = 1.3, cdf = False,kmaxnum = False)
collapse(data2, num, ms[2], green,scale = 1.3, cdf = False,kmaxnum = False)
collapse(data3, num, ms[3], purple,scale = 1.3, cdf = False,kmaxnum = False)
collapse(data4, num, ms[4], pink,scale = 1.3, cdf = False,kmaxnum = False)
collapse(data5, num, ms[5], yellow,scale = 1.3, cdf = False,kmaxnum = False)
collapse(data6, num, ms[6], grey,scale = 1.3, cdf = False,kmaxnum = False)

collapse(datan0, nums[0], m, 'red',scale = 1.3, cdf = False,kmaxnum = False)
collapse(datan1, nums[1], m, 'blue',scale = 1.3, cdf = False,kmaxnum = False)
collapse(datan2, nums[2], m, 'green',scale = 1.3, cdf = False,kmaxnum = False)
collapse(datan3, nums[3], m,'purple',scale = 1.3, cdf = False,kmaxnum = False)
collapse(datan4, nums[4], m, 'pink',scale = 1.3, cdf = False,kmaxnum = False)
collapse(datan5, nums[5], m, 'yellow',scale = 1.3, cdf = False,kmaxnum = False)
collapse(datan6, nums[6], m, 'grey',scale = 1.3, cdf = False,kmaxnum = False)

#plt.legend()
#%%
collapse(data0, num, ms[0], red,scale = 1.2, cdf = True)
collapse(data1, num, ms[1], blue,scale = 1.2, cdf = True)
collapse(data2, num, ms[2], green,scale = 1.2, cdf = True)
collapse(data3, num, ms[3], purple,scale = 1.2, cdf = True)
collapse(data4, num, ms[4], pink,scale = 1.2, cdf = True)
collapse(data5, num, ms[5], yellow,scale = 1.2, cdf = True)
collapse(data6, num, ms[6], grey,scale = 1.2, cdf = True)

collapse(datan0, nums[0], m, 'red',scale = 1.3, cdf = True)
collapse(datan1, nums[1], m, 'blue',scale = 1.3, cdf = True)
collapse(datan2, nums[2], m, 'green',scale = 1.3, cdf = True)
collapse(datan3, nums[3], m,'purple',scale = 1.3, cdf = True)
collapse(datan4, nums[4], m, 'pink',scale = 1.3, cdf = True)
collapse(datan5, nums[5], m, 'yellow',scale = 1.3, cdf = True)
collapse(datan6, nums[6], m, 'grey',scale = 1.3, cdf = True)

#plt.legend()
#%%
collapse(data0, num, ms[0], red,scale = 1.2, cdf = True,kmaxnum = False)
collapse(data1, num, ms[1], blue,scale = 1.2, cdf = True,kmaxnum = False)
collapse(data2, num, ms[2], green,scale = 1.2, cdf = True,kmaxnum = False)
collapse(data3, num, ms[3], purple,scale = 1.2, cdf = True,kmaxnum = False)
collapse(data4, num, ms[4], pink,scale = 1.2, cdf = True,kmaxnum = False)
collapse(data5, num, ms[5], yellow,scale = 1.2, cdf = True,kmaxnum = False)
collapse(data6, num, ms[6], grey,scale = 1.2, cdf = True,kmaxnum = False)

collapse(datan0, nums[0], m, 'red',scale = 1.3, cdf = True,kmaxnum = False)
collapse(datan1, nums[1], m, 'blue',scale = 1.3, cdf = True,kmaxnum = False)
collapse(datan2, nums[2], m, 'green',scale = 1.3, cdf = True,kmaxnum = False)
collapse(datan3, nums[3], m,'purple',scale = 1.3, cdf = True,kmaxnum = False)
collapse(datan4, nums[4], m, 'pink',scale = 1.3, cdf = True,kmaxnum = False)
collapse(datan5, nums[5], m, 'yellow',scale = 1.3, cdf = True,kmaxnum = False)
collapse(datan6, nums[6], m, 'grey',scale = 1.3, cdf = True,kmaxnum = False)

#plt.legend()












    
    