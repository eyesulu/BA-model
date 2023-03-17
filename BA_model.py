#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:39:39 2023

@author: aisulu
"""
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
plt.style.use('ggplot')
plt.rcParams["font.family"] = "Times New Roman"
red = list(plt.rcParams['axes.prop_cycle'])[0]['color']
blue = list(plt.rcParams['axes.prop_cycle'])[1]['color']
purple = list(plt.rcParams['axes.prop_cycle'])[2]['color']
grey = list(plt.rcParams['axes.prop_cycle'])[3]['color']
yellow = list(plt.rcParams['axes.prop_cycle'])[4]['color']
green = list(plt.rcParams['axes.prop_cycle'])[5]['color']
pink = list(plt.rcParams['axes.prop_cycle'])[6]['color']

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
def power_func(k, A, b):
    return A*k**b
def power_func_k(k, A):
    return A/(k*(k+1)*(k+2))
def yule(k, r, A):
    prob = A * np.exp(sc.gammaln(k-r)-sc.gammaln(k+1))
    return prob
def survivor_function(x, C, beta):
    return C/(beta-1) * x**(-beta-1)
def pareto(x, xm, alpha):
    return (xm/x)**alpha
                            
class BA_model:
    def __init__(self, N, m, edges =[]):
        self._N = N
        self._m = m
        self._nodes = np.arange(self._N)
        if edges == []:
            if self._m > self._N+1:
                raise Exception('Choose a higher value of N')
            else:
                self._edges = []
                for node in self._nodes:
                    nodes = self._nodes
                    edge1 = node
                    edges2 = np.delete(nodes, node)
                    for edge in edges2:
                        self._edges.append([edge1, edge])
        else:
            self._edges = edges
        edges_for_flat = np.array(self._edges)
        self._edges_flat = edges_for_flat.flatten()
        self._degree_dict = dict(Counter(self._edges_flat))
        self._degrees = list(self._degree_dict.values())
    def add_node(self):
        prob_select = self._degrees/np.sum(self._degrees)
        edge1 = max(self._nodes)+1
        edge2 = np.random.choice(self._nodes,self._m, replace = False, p = prob_select)
        new_edges = []
        for i in edge2:
            new_edges.append([edge1, i])
            self._degrees[i]+=1
        self._degrees = np.append(self._degrees, self._m)
        self._edges = np.concatenate((self._edges, new_edges))
        self._nodes = np.append (self._nodes,max(self._nodes)+1)

    def add_nodes(self, num):
        for i in range(num):
            print (i)
            self.add_node()
    def plot(self):
        G = nx.Graph()
        G.add_edges_from(self._edges)
        nx.draw(G, pos=nx.circular_layout(G), with_labels = True)
        plt.title('Graph')
    def plot_fat_tail(self, num, scale, show_data = True):
        self.add_nodes(num)
        x, y = getting_prob(self._degrees)
        x_binned, y_binned = lg.logbin(self._degrees, scale = scale)
        xs = np.linspace (min(self._degrees), max(self._degrees), 1000)
        po,po_cov=sp.optimize.curve_fit(power_func_k,x_binned, y_binned)
        plt.loglog (xs, power_func_k (xs, po), linewidth = 0.8, label ='Power function')
        po2, po_cov2 = sp.optimize.curve_fit(pareto,x_binned, y_binned)
        plt.loglog(xs, pareto(xs, po2[0], po2[1]), label = 'Pareto function',linewidth = 0.8 )
        po3,po_cov3=sp.optimize.curve_fit(yule,x_binned, y_binned)
        plt.loglog(xs, yule(xs, po3[0], po3[1]), label = 'Yule distribution')
        if show_data == True:
            plt.loglog(x, y, 'x', label = 'Unbinned', color = 'grey')
            plt.loglog(x_binned, y_binned, '.', label = 'Log-binned', color = 'black')
        plt.legend()
        plt.grid(False)
        plt.xlabel(r'ln (k)')
        plt.ylabel(r'ln (P(k))')
        plt.title(r'Degree distribution for N = %i' %num)
        statistic_power, p_value_power = stats.kstest(y_binned,power_func_k (x_binned, po) )
        statistic_pareto, p_value_pareto = stats.kstest(y_binned,pareto(x_binned, po2[0], po2[1]) )
        statistic_yule, p_value_yule = stats.kstest(y_binned,yule(x_binned, po3[0], po3[1]) )
        print ('The p value for the power-law fuction = %.3e' %p_value_power)
        print ('The p value for the Pareto fuction = %.3e' %p_value_pareto)
        print ('The p value for the Yule fuction = %.3e' %p_value_yule)
    def plot_hist(self, N):
        self.add_nodes(N)
        plt.hist(self._degrees, bins = 25)
    def plot_cdf(self, num):
        self.add_nodes(num)
        #x, y = getting_prob(self._degrees)
        x_cdf, y_cdf = getting_cum_prob(self._degrees)
        #plt.loglog(x, y, 'x', label = 'Unbinned', color = 'grey')
        plt.loglog(x_cdf,y_cdf, 'x', label = 'CDF', color = 'black')
        xs = np.linspace (min(x_cdf), max(x_cdf), 1000)
        #po,po_cov=sp.optimize.curve_fit(power_func_k,x_cdf, y_cdf)
        po2, po_cov2 = sp.optimize.curve_fit(pareto,x_cdf, y_cdf)
        po3,po_cov3=sp.optimize.curve_fit(yule,x_cdf, y_cdf)
        plt.loglog(xs, yule(xs, po3[0], po3[1]), label = 'Yule distribution', linewidth = 0.8)
        #plt.loglog (xs, power_func_k (xs, po), label ='Power function')
        plt.loglog(xs, pareto(xs, po2[0], po2[1]), label = 'Pareto function',linewidth = 0.8 )
        print (po3)
        print (x_cdf, y_cdf)
        #plt.loglog(x_cdf,yule(np.array(x_cdf), po3[0], po3[1]) , label = 'test', 'x')
        plt.legend()
        #statistic_power, p_value_power = stats.kstest(y_cdf,power_func_k (x_cdf, po) )
        statistic_pareto, p_value_pareto = stats.kstest(y_cdf,pareto(np.array(x_cdf), po2[0], po2[1]) )
        statistic_yule, p_value_yule = stats.kstest(y_cdf,yule(np.array(x_cdf), po3[0], po3[1]) )
        #print ('The p value for the power-law fuction = %.3e' %p_value_power)
        print ('The p value for the Pareto fuction = %.5f' %p_value_pareto)
        print ('The p value for the Yule fuction = %.5f' %p_value_yule)
    def degrees(self):
        return self._degrees
    def edges(self):
        return self._edges


'''
add_nodes(1000) Completed in 5.8 seconds
'''





        