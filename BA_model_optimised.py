#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:52:10 2023

@author: aisulu
"""

import numpy as np
import random
from collections import Counter

class BA_model_opt:
    def __init__(self, m, model = 'PA', r = 0, multi = False):
        self._m = m
        self._N = self._m + 1
        self._ends = []
        self._model = model
        self._r = r
        self._multi = multi
        for node in range(self._N):
            for i in range (self._N):
                if node != list(np.arange(self._N))[i]:
                    self._ends.append(node)
        self._degree_dict = dict(Counter(self._ends))
        self._degrees = list(self._degree_dict.values())
        self._nodes = list(np.arange(self._N))
    def ends(self):
        return self._ends
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
        if self._model == 'EV':
            self._degrees.append(self._r)
            new_ends = []
            new_end_count = 0
            new_end_pairs = []
            while new_end_count < self._r:
                new_end = random.choice(self._nodes)
                if new_end not in new_ends:
                    new_ends.append(new_end)
                    new_end_count += 1
            while new_end_count < self._m:
                new_end1 = random.choice(self._ends)
                new_end2 = random.choice(self._ends)
                if [new_end1, new_end2] not in new_end_pairs:
                    new_end_pairs.append([new_end1, new_end2])
                    new_end_count += 1
            for i in new_ends:
                self._ends.append(i)
                self._ends.append(self._N)
                self._degrees[i] += 1
            for pair in new_end_pairs:
                self._ends.append(pair[0])
                self._ends.append(pair[1])
                self._degrees[pair[0]] += 1
                self._degrees[pair[1]] += 1
            self._N += 1
        self._nodes.append(self._N-1)
    def add_nodes(self, num):
        for i in range (num):
            self.add_node()
        print ('Adding nodes complete for m = %i, N = %i' %(self._m, num))
    def nodes(self):
        return self._nodes
        

                
        
        
            
            
    
        
        
        
        
        