# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:58:44 2016

@author: aitor
"""
from __future__ import division
import csv


DATASET = './dataset/action_dataset.csv'
TEXT = './dataset/actions.text'
BY_PERIOD = False

print 'Starting...'
with open(DATASET, 'rb') as csvfile:
    print 'Processing:', DATASET
    reader = csv.reader(csvfile)
    phrases = []
    current_phrase = []
    current_activity = ''
    i = 0
    for row in reader:
        i += 1
        if i == 1:
            continue        
        date = row[0]
        action = row[1]
        activity = row[2]
        if BY_PERIOD:
            hour = date.split(' ')[1].split(':')[0]
            minute = date.split(' ')[1].split(':')[1]
            if int(minute) > 30:
                minute = '45'
            else:
                minute = '15'
            period = hour+minute
            action = action+period
        
        if activity == current_activity:
            current_phrase.append(action)
        else:
            phrases.append(current_phrase)
            current_activity = activity
            current_phrase = [action]  
        if i % 100000 == 0:
            print 'Actions processed:', i

print 'Total phrases:', len(phrases) 
print 'Saving text file...'
total = 0
length = 0
with open(TEXT, 'w') as textfile: 
    for phrase in phrases:
        total += 1
        length += len(phrase)
        phrase_text = ' '.join(phrase)
        phrase_text = phrase_text + '\n'
        textfile.write(phrase_text)         
print 'Average activity length:',  length / total        
        
print 'FIN'
