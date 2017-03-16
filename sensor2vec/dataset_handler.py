# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:07:36 2016

@author: aitor
"""
from collections import Counter
import csv
import json

from gensim.models import Word2Vec
import numpy


# Dataset generated with the synthetic generator
DATASET_ACTION = 'action_dataset.csv'
# Kasteren dataset
DATASET_KASTEREN = 'kasteren_dataset.csv'
# Text file used to create the action vectors with word2vec
ACTION_TEXT = 'actions.txt'
# List of unique activities in the dataset
UNIQUE_ACTIVITIES = 'unique_activities.json'
# List of unique actions in the dataset
UNIQUE_ACTIONS = 'unique_actions.json'
# Word2vec model generated with gensim
ACTIONS_MODEL = 'actions.model'
# Vector values for each action
ACTIONS_VECTORS = 'actions_vectors.json'
# File with the activities ordered
ACTIVITIES_ORDERED = 'activities.json'
# Dataset with vectors but without the action timestamps
DATASET_NO_TIME = 'dataset_no_time.json'
# Dataset with vectors but without the action timestamps, 2 channels
DATASET_NO_TIME_2_CHANNELS = 'dataset_no_time_2_channels.json'

# When there is no activity
NONE = 'None'
# Separator for the text file
SEP = ' '
# Maximun number of actions in an activity
ACTIVITY_MAX_LENGHT = 32
# word2vec dimensions for an action
ACTION_MAX_LENGHT = 50

DATASET = DATASET_KASTEREN

# Generates the text file from the csv
def process_csv(none=False):
    actions = ''    
    activities_set = set()
    actions_set = set()
    with open(DATASET, 'rb') as csvfile:
        print 'Processing:', DATASET
        reader = csv.reader(csvfile)        
        i = 0
        for row in reader:
            i += 1
            if i == 1:
                continue        
            action = row[1]
            activity = row[2]
            if none:
                activities_set.add(activity)      
                actions_set.add(action)
            if activity != NONE and not none:
                actions += action + SEP
                activities_set.add(activity)
                actions_set.add(action)
            if i % 10000 == 0:
                print '  -Actions processed:', i
        print 'Total actions processed:', i
    
    with open(ACTION_TEXT, 'w') as textfile: 
        textfile.write(actions)     
    json.dump(list(activities_set), open(UNIQUE_ACTIVITIES, 'w'))
    json.dump(list(actions_set), open(UNIQUE_ACTIONS, 'w'))
    print 'Text file saved'

# creates a json file with the action vectors from the gensim model
def create_vector_file():
    print 'Creating the vector file...'
    actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    print 'Total unique actions:', len(actions)
    model = Word2Vec.load(ACTIONS_MODEL)
    actions_vectors = {}
    for action in actions:
        try:
            action_values = model[action].tolist()
        except:
            action_values = [0.0] * ACTION_MAX_LENGHT
        actions_vectors[action] = action_values
     
    json.dump(actions_vectors, open(ACTIONS_VECTORS, 'w'), indent=2)
    print 'Saved action vectors'

# Processes the csv and orders the activities in a json    
def order_activities(none = False):
    with open(DATASET, 'rb') as csvfile:
        print 'Processing:', DATASET
        reader = csv.reader(csvfile)        
        i = 0        
        current_activity = {
            'name':'',
            'actions': []
        }
        activities = []
        for row in reader:
            i += 1
            if i == 1:
                continue        
            date = row[0]            
            action = row[1]
            activity = row[2]
            if (activity != NONE and not none) or (none):
                if activity == current_activity['name']:
                    action_data = {
                        'action':action,
                        'date':date                
                    }
                    current_activity['actions'].append(action_data)
                else:
                    activities.append(current_activity)
                    current_activity = {
                        'name':activity,
                        'actions': []
                    }
                    action_data = {
                        'action':action,
                        'date':date                
                    }
                    current_activity['actions'].append(action_data)
            if i % 10000 == 0:
                print 'Actions processed:', i
        json.dump(activities, open(ACTIVITIES_ORDERED, 'w'), indent=1)
    print 'Ordered activities'
    
def median(lst):
    return numpy.median(numpy.array(lst))
 
# Statistics about the activities   
def calculate_statistics():
    print 'Calculating statistics'
    activities = json.load(open(ACTIVITIES_ORDERED, 'r'))
    total_activities = len(activities)
    print 'Total activities:', total_activities
    action_lengths = []
    for activity in activities:
        action_lengths.append(len(activity['actions']))
    print 'Avg activity lenght:', sum(action_lengths)/total_activities
    print 'Median activity lenght:', median(action_lengths)
    print 'Longest activity:', max(action_lengths)
    print 'Shortest activity:', min(action_lengths)
    distribution = Counter(action_lengths)
    print 'Distribution:', json.dumps(distribution, indent=2)
    
def create_vector_dataset_no_time():
    print 'Creating dataset...'
    dataset = []
    action_vectors = json.load(open(ACTIONS_VECTORS, 'r'))
    unique_activities = json.load(open(UNIQUE_ACTIVITIES, 'r'))
    activities = json.load(open(ACTIVITIES_ORDERED, 'r'))
    for activity in activities:
        one_hot_activity = [0.0] * len(unique_activities)
        activity_index = unique_activities.index(activity['name'])
        one_hot_activity[activity_index] = 1.0
        training_example = {
            'activity' : one_hot_activity,
            'actions' : []         
        }
        for action in activity['actions']:
            action_values = action_vectors[action['action']]
            training_example['actions'].append(action_values)

        # Padding        
        if len(training_example['actions']) < ACTIVITY_MAX_LENGHT:
            for i in range(ACTIVITY_MAX_LENGHT - len(training_example['actions'])):
              training_example['actions'].append([0.0] * ACTION_MAX_LENGHT)
              
        dataset.append(training_example)
    print 'Writing file'
    json.dump(dataset, open(DATASET_NO_TIME,'w'))
    print 'Created dataset'
    
def create_vector_dataset_no_time_2_channels():
    single_dataset = json.load(open(DATASET_NO_TIME, 'r'))
    dataset = []
    previous_activity = None
    for activity in single_dataset:
        if previous_activity == None:
            previous_activity = activity
        else:
            training_example = {
                'activity' : activity['activity'],
                'actions' : activity['actions'], 
                'previous_actions': previous_activity['actions']
            }                       
            dataset.append(training_example)
            previous_activity = activity
    print 'Writing file'
    json.dump(dataset, open(DATASET_NO_TIME_2_CHANNELS,'w'))
    print 'Created dataset'           
        
    
    
if __name__ == '__main__':
    print 'Start...'
#    process_csv(True)
#    create_vector_file()
#    order_activities(True)
#    calculate_statistics()
#    create_vector_dataset_no_time()
    create_vector_dataset_no_time_2_channels()
    print 'Fin'



            


