# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:34:49 2016

@author: aitor
"""
total = 0

with open('editoriales.txt', 'r') as f:
    for l in f:
        words = len(l.split(' '))
        total += words            

print total