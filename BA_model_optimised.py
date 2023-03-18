#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:52:10 2023

@author: aisulu
"""

import numpy as np
import random
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

class BA_model_opt:
    def __init__(self, m, model = 'PA', r = 0, multi = False):
        self._m = m
        self._model = model
        self._r = r
        self._multi = multi
        if model == 'PA' or 'RA':
            self._ends = []
            self._N = self._m + 1
            self._nodes = list(np.arange(self._N))
            for node in range(self._N):
                for i in range (self._N):
                    if node != list(np.arange(self._N))[i]:
                        self._ends.append(node)
        if model == 'EV':
            self._ends = []
            self._edges = []
            self._N = 2*self._m + 1
            self._nodes = list(np.arange(self._N))
            for node in self._nodes:
                for i in self._nodes[:self._m+1]:
                    if node != i:
                        if [node, i] and [i, node] not in self._edges:
                            self._edges.append([node, i])
                            self._ends.append(node)
                            self._ends.append(i)
        self._degree_dict = dict(Counter(self._ends))
        self._degrees = list(self._degree_dict.values())
    def ends(self):
        return self._ends
    def edges(self):
        return self._edges
    def degrees(self):
        return self._degrees
    def add_node(self):
        if self._model == 'PA':
            if self._multi == False:
                self._degrees.append(0)
                new_ends = []
                new_end_count = 0
                while new_end_count < self._m:
                    new_end = random.choice(self._ends)
                    if new_end not in new_ends:
                        new_ends.append(new_end)
                        new_end_count += 1
                for i in new_ends:
                    self._ends.append(i)
                    self._ends.append(self._N)
                    self._degrees[self._N] += 1
                    self._degrees[i] += 1
                self._N += 1
            if self._multi == True:
                self._degrees.append(0)
                new_ends = []
                new_end_count = 0
                while new_end_count < self._m:
                    new_end = random.choice(self._ends)
                    new_ends.append(new_end)
                    new_end_count += 1
                for i in new_ends:
                    self._ends.append(i)
                    self._ends.append(self._N)
                    self._degrees[self._N] += 1
                    self._degrees[i] += 1
                self._N += 1
        if self._model == 'RA':
            if self._multi == False:
                self._degrees.append(0)
                new_ends = []
                new_end_count = 0
                while new_end_count < self._m:
                    new_end = random.choice(self._nodes)
                    if new_end not in new_ends:
                        new_ends.append(new_end)
                        new_end_count += 1
                for i in new_ends:
                    self._ends.append(i)
                    self._ends.append(self._N)
                    self._degrees[self._N] += 1
                    self._degrees[i] += 1
                self._N += 1
            if self._multi == True:
                self._degrees.append(0)
                new_ends = []
                new_end_count = 0
                while new_end_count < self._m:
                    new_end = random.choice(self._nodes)
                    new_ends.append(new_end)
                    new_end_count += 1
                for i in new_ends:
                    self._ends.append(i)
                    self._ends.append(self._N)
                    self._degrees[self._N] += 1
                    self._degrees[i] += 1
        if self._model == 'EV':
            if self._multi == False:
                self._degrees.append(self._r)
                new_ends = []
                new_end_pairs = []
                while len(new_ends) < self._r:
                    new_end = random.choice(self._nodes)
                    if new_end not in new_ends:
                        new_ends.append(new_end)
                while len(new_end_pairs) < self._m-self._r:
                    new_end1 = random.choice(self._ends)
                    new_end2 = random.choice(self._ends)
                    if new_end1 != new_end2:
                        if [new_end1, new_end2] not in new_end_pairs and [new_end2, new_end2] not in new_end_pairs:
                            if [new_end1, new_end2] not in self._edges and [new_end2, new_end1] not in self._edges:
                                
                                new_end_pairs.append([new_end1, new_end2])
                for i in new_ends:
                    self._ends.append(i)
                    self._ends.append(self._N)
                    self._degrees[i] += 1
                    self._edges.append([self._N, i])
                for pair in new_end_pairs:
                    self._ends.append(pair[0])
                    self._ends.append(pair[1])
                    self._degrees[pair[0]] += 1
                    self._degrees[pair[1]] += 1
                    self._edges.append([pair[0], pair[1]])
                self._N += 1
        self._nodes.append(self._N-1)
    def add_nodes(self, num):
        for i in range (num):
            self.add_node()
        print ('Adding nodes complete for m = %i, N = %i' %(self._m, num))
    def nodes(self):
        return self._nodes
    def plot(self):
        G = nx.Graph()
        G.add_edges_from(self._edges)
        nx.draw(G, pos=nx.circular_layout(G), with_labels = True)
        plt.title('Graph')
            
            
    
        
        
        
        
        
