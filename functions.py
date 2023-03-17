#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 19:47:25 2023

@author: aisulu
"""

import numpy as np
import matplotlib.pyplot as plt
import logbin as lg
from collections import Counter
import math
import scipy as sp
from scipy.optimize import curve_fit
import scipy.special as sc
from scipy import stats
from matplotlib.lines import Line2D
def power_func(k, A, b):
    return A*k**b
def getting_prob (data):
    #the counter method was taken from :https://realpython.com/python-counter/
    #it creates a dict of different values in data and their frequency 
    counter = {}
    for i in data:
        counter[i] = counter.get(i, 0) + 1
    values = []
    probs = []
    for i in counter:
        values.append (i)
        probs.append (counter[i])
    probs = np.divide (probs, len(data))
    return values, probs
def getting_cum_prob(data):
    counter = {}
    for i in data:
        counter[i] = counter.get(i, 0) + 1
    values = []
    probs = []
    counter = dict(sorted(counter.items()))
    for i in counter:
        values.append (i)
        probs.append (counter[i])
    y_cdf = []
    for i in range(len(probs)):
        ys = []
        ys.append(probs[i:])
        y_cdf.append(np.sum(ys))
    y_cdf = np.divide (y_cdf, len(data))
    return values, y_cdf
def theoretical_dist_PA(k, m, cdf = False):
    if cdf == False:
        prob = 2*m*(m+1) / (k*(k+1)*(k+3))
        return prob
    else:
        prob = m*(m+1) / (k*(k+1))
        return prob
def theoretical_dist_RA(k, m, cdf = False):
    if cdf == False:
        prob = np.exp((k-m)* np.log(m) - ((1+k-m)*np.log(1+m)))
        return prob
    else:
        prob = -1/((m**m)*(1+m)**(1-m)) * np.power( (m/(1+m)),k) / (m/(1+m) -1) 
        return prob
def theoretical_dist_EV(k, m, r, cdf = False, risth = False):
    if cdf == False:
        if risth == True:
            A = 3 * m * (3*m+2) / 2
            return A/((k+m)*(k+m+1)*(k+m+2))
        else:
            coef = (2*r*m-r**2)/(m+2*r*m-r**2)
            suma = 0
            for i in range(int(10e3)*r):
                print (i)
                suma += np.exp(sc.gammaln(r+1+i+m*r/(m-r))-sc.gammaln(r+1+i+1+m*r/(m-r)+m/(m-r)))
            A = coef/suma
            return A*np.exp(sc.gammaln(k+m*r/(m-r))-sc.gammaln(k+1+m*r/(m-r)+m/(m-r)))
    else:
        return 0
def plot_m_dist (data, scale, num, m, color,show_title = True, cdf = False, model = 'PA', r = 0):
    xs = np.linspace (min(data), max(data), 1000)
    if model == 'PA':
        if cdf == False:
            x, y = getting_prob(data)
            x_binned, y_binned = lg.logbin(data, scale = scale)
            y_theory = theoretical_dist_PA(xs, m)
            plt.loglog(x_binned, y_binned,'.', color = color, alpha = 0.5)
            plt.ylabel(r'PDF')
        else:
            x, y = getting_cum_prob(data)
            y_theory = theoretical_dist_PA(xs, m, cdf = True)
            plt.loglog(x, y,'.', color = color, alpha = 0.5)
        plt.loglog(xs, y_theory,'--',linewidth = 0.8, label =r'Theoretical dist. for $m = %i$' %m, color = color) 
        plt.legend()
        plt.grid(False)
        plt.xlabel(r'$ k$')
        plt.ylabel(r'CCDF')
        if show_title == True:
            plt.title(r'Degree distribution for N = %i, scale = %.2f, PA model' %(num, scale))
    if model == 'RA':
        if cdf == False:
            x, y = getting_prob(data)
            x_binned, y_binned = lg.logbin(data, scale = scale)
            y_theory = theoretical_dist_RA(xs, m)
            plt.loglog(x_binned, y_binned,'.', color = color, alpha = 0.5)
            plt.ylabel(r'PDF')
        else:
            x, y = getting_cum_prob(data)
            y_theory = theoretical_dist_RA(xs, m, cdf = True)
            plt.loglog(x, y,'.', color = color, alpha = 0.5)
            plt.ylabel(r'CCDF')
        plt.loglog(xs, y_theory,'--',linewidth = 0.8, label =r'Theoretical dist. for $m = %i$' %m, color = color) 
        plt.legend()
        plt.grid(False)
        plt.xlabel(r'$k$')
        if show_title == True:
            plt.title(r'Degree distribution for N = %i, scale = %.2f, RA model' %(num, scale))
    if model == 'EV':
        if cdf == False:
            x, y = getting_prob(data)
            x_binned, y_binned = lg.logbin(data, scale = scale)
            y_theory = theoretical_dist_EV(xs, m, r)
            plt.loglog(x_binned, y_binned,'.', color = color, alpha = 0.5)
            plt.loglog(xs, y_theory,'--',linewidth = 0.8, label =r'Theoretical dist. for $m = %i$' %m, color = color) 
            plt.xlabel(r'$k$')
            plt.ylabel(r'CCDF')
            if show_title == True:
                plt.title(r'Degree distribution for N = %i, scale = %.2f, EV model' %(num, scale))
def plot_N_dist(data, scale, num, m, color,show_title = True, plot_th = False, cdf = False, model = 'PA', r = 0):
    xs = np.linspace (min(data), max(data), 1000)
    if model == 'PA':
        if cdf == False:
            x, y = getting_prob(data)
            x_binned, y_binned = lg.logbin(data, scale = scale)
            plt.loglog(x_binned, y_binned,'x', color = color, label = r'N = %i' %num, alpha = 0.9)
            if plot_th == True:
                y_theory = theoretical_dist_PA(xs, m)
                plt.loglog(xs, y_theory,'--',linewidth = 0.8, label =r'Theoretical dist. for $m = %i$' %m, color = 'black') 
            plt.legend()
            plt.grid(False)
            plt.xlabel(r'$k$')
            plt.ylabel(r'PDF')
            if show_title == True:
                plt.title(r'Degree distribution for m = %i, scale = %.2f, PA model' %(m, scale))
        else:
            x, y = getting_cum_prob(data)
            plt.loglog(x, y,'x', color = color, label = r'N = %i' %num, alpha = 0.9)
            if plot_th == True:
                xs = np.linspace (min(data), max(data), 1000)
                y_theory = theoretical_dist_PA(xs, m, cdf = True)
                plt.loglog(xs, y_theory,'--',linewidth = 0.8, label =r'Theoretical dist. for $m = %i$' %m, color = 'black') 
            plt.legend()
            plt.grid(False)
            plt.xlabel(r'$k$')
            plt.ylabel(r'CCDF')
            if show_title == True:
                plt.title(r'Degree distribution for m = %i, scale = %.2f, PA model' %(m, scale))
    if model == 'RA':
        if cdf == False:
            x, y = getting_prob(data)
            x_binned, y_binned = lg.logbin(data, scale = scale)
            xs = np.linspace (min(data), max(data), 1000)
            plt.loglog(x_binned, y_binned,'.', color = color, alpha = 0.5, label = r'N = %i' %num)
            if plot_th == True:
                y_theory = theoretical_dist_RA(xs, m, cdf = False)
                plt.loglog(xs, y_theory,'--',linewidth = 0.8, label =r'Theoretical dist. for $m = %i$' %m, color = 'black') 
            plt.legend()
            plt.grid(False)
            plt.xlabel(r'$ k$')
            plt.ylabel(r'PDF')
            if show_title == True:
                plt.title(r'Degree distribution for m = %i, scale = %.2f, RA model' %(m, scale))
        else:
            x, y = getting_cum_prob(data)
            plt.loglog(x, y,'.', color = color, label = r'N = %i' %num, alpha = 0.9)
            if plot_th == True:
                y_theory = theoretical_dist_RA(xs, m, cdf = True)
                plt.loglog(xs, y_theory,'--',linewidth = 0.8, label =r'Theoretical dist. for $m = %i$' %m, color = 'black') 
            plt.legend()
            plt.grid(False)
            plt.xlabel(r'$k$')
            plt.ylabel(r'CCDF')
            if show_title == True:
                plt.title(r'Degree distribution for m = %i, scale = %.2f, RA model' %(m, scale))
    if model == 'EV':
        if cdf == False:
            x, y = getting_prob(data)
            x_binned, y_binned = lg.logbin(data, scale = scale)
            plt.loglog(x_binned, y_binned,'.', color = color, alpha = 0.5)
            if plot_th == True:
                y_theory = theoretical_dist_EV(xs, m, r)
                plt.loglog(xs, y_theory,'--',linewidth = 0.8, label =r'Theoretical dist. for $m = %i$' %m, color = 'black') 
            plt.legend()
            plt.grid(False)
            plt.xlabel(r'$ k$')
            plt.ylabel(r'PDF')
            if show_title == True:
                plt.title(r'Degree distribution for N = %i, scale = %.2f, EV model' %(num, scale))
'''
Add CCDF for EV
'''
def stat_test(data, m, num, scale, sig, cdf = False, short_print = False, model = 'PA', r = 0):
    if model == 'PA':
        if cdf == False:
            x_binned, y_binned = lg.logbin(data, scale = scale)
            statistic, p_value = stats.kstest(y_binned, theoretical_dist_PA(x_binned, m) )
        else:
            x, y = getting_cum_prob(data)
            x = np.array(x)
            statistic, p_value = stats.kstest(y, theoretical_dist_PA(x, m, cdf = True) )
    if model == 'RA':
        if cdf == False:
            x_binned, y_binned = lg.logbin(data, scale = scale)
            statistic, p_value = stats.kstest(y_binned, theoretical_dist_RA(x_binned, m) )
        else:
            x, y = getting_cum_prob(data)
            statistic, p_value = stats.kstest(y, theoretical_dist_RA(x, m, cdf = True) )
    if model == 'EV':
        if cdf == False:
            x_binned, y_binned = lg.logbin(data, scale = scale)
            statistic, p_value = stats.kstest(y_binned, theoretical_dist_EV(x_binned, m, r) )
        '''
        Add CCDF for EV
        '''
    print ('Statistic = %3f, p value = %.3f' %(statistic, p_value))
def exp_kmax(Ns, m, listofn = True, model = 'PA'):
    if model == 'PA':
        if listofn == True:
            answer = []
            for N in Ns:
                k_max = (-1+np.sqrt(1+4*N*m*(m+1)))/2
                answer.append(k_max)
            return answer
        else:
            return (-1+np.sqrt(1+4*Ns*m*(m+1)))/2
    if model == 'RA':
        if listofn == True:
            answer = []
            for N in Ns:
                k_max = m - np.log(N)/(np.log(m)-np.log(m+1))
                answer.append(k_max)
            return answer
        else:
            return m - np.log(Ns)/(np.log(m)-np.log(m+1))
'''
Add kmax for EV
'''

def collapse(data, num, m, color ,scale, cdf = False, model = 'PA', r = 0, kmaxnum = True):
    if model == 'PA':
        if cdf == False:
            x_binned, y_binned = lg.logbin(data, scale = scale)
            if kmaxnum == True:
                k1 = exp_kmax(num, m, listofn = False)
            else:
                k1 = np.max(x_binned)
            x = x_binned / k1
            y = np.divide(y_binned,theoretical_dist_PA(x_binned, m, cdf = False))
            plt.loglog(x, y, '-', color=color, label = r'$N = %i, m = %.i$' %(num, m) , alpha = 0.5)
            plt.xlabel (r'$k / k_1$')
            plt.ylabel (r'$p (k) / p_{\infty} (k)$')
        else:
            x_cdf, y_cdf = getting_cum_prob(data)
            if kmaxnum == True:
                k1 = exp_kmax(num, m, listofn = False)
            else:
                k1 = np.max(x_cdf)
            x = x_cdf / k1
            y = np.divide(y_cdf,theoretical_dist_PA(np.array(x_cdf), m, cdf = True))
            plt.loglog(x, y, '-', color=color, label = r'$N = %i, m = %.i$' %(num, m), alpha = 0.5)
            plt.xlabel (r'$k / k_1$')
            plt.ylabel (r'$p_{>} (k) / p_{> , \infty} (k)$')
    if model == 'RA':
        if cdf == False:
            x_binned, y_binned = lg.logbin(data, scale = scale)
            if kmaxnum == True:
                k1 = exp_kmax(num, m, listofn = False,model = 'RA')
            else:
                k1 = np.max(x_binned)
            x = x_binned / k1
            y = np.divide(y_binned,theoretical_dist_RA(x_binned, m, cdf = False))
            plt.loglog(x, y, '.', color=color, label = r'$N = %i, m = %.i$' %(num, m) , alpha = 0.5)
            plt.xlabel (r'$k / k_1$')
            plt.ylabel (r'$p (k) / p_{\infty} (k)$')
        else:
            x_cdf, y_cdf = getting_cum_prob(data)
            if kmaxnum == True:
                k1 = exp_kmax(num, m, listofn = False,model = 'RA')
            else:
                k1 = np.max(x_cdf)
            x = x_cdf / k1
            y = np.divide(y_cdf,theoretical_dist_RA(np.array(x_cdf), m, cdf = True))
            plt.loglog(x, y, '.', color=color, label = r'$N = %i, m = %.i$' %(num, m), alpha = 0.5)
            plt.xlabel (r'$k / k_1$')
            plt.ylabel (r'$p_{>} (k) / p_{> , \infty} (k)$')
              
              
       

    
    
    
    
    
    
    
    
        
            
    
    
            
    
    
    
    
  
    
    
    
    
    
    
    
    
   
            
















