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

#base directory
BASE_DIR = './kasteren_dataset/'
# Dataset generated with the synthetic generator
DATASET_ACTION = BASE_DIR + 'action_dataset.csv'
# Kasteren dataset
DATASET_KASTEREN = BASE_DIR + 'kasteren_dataset.csv'
# Kasteren dataset reduced
DATASET_KASTEREN_REDUCED = BASE_DIR + 'base_kasteren_reduced.csv'
# Output dataset with actions transformed with time periods
DATASET_ACTION_PERIODS = BASE_DIR + 'kasteren_action_periods.csv'
# Text file used to create the action vectors with word2vec
ACTION_TEXT = BASE_DIR + 'actions.txt'
# Text file used to create the action vectors including time with word2vec
ACTION_TIME_TEXT = BASE_DIR + 'actions_time.txt'
# List of unique activities in the dataset
UNIQUE_ACTIVITIES = BASE_DIR + 'unique_activities.json'
# List of unique actions in the dataset
UNIQUE_ACTIONS = BASE_DIR + 'unique_actions.json'
# List of unique actions in the dataset when using time periods
UNIQUE_TIME_ACTIONS = BASE_DIR + 'unique_time_actions.json'
# Word2vec model generated with gensim
ACTIONS_MODEL = BASE_DIR + 'actions.model'
# Vector values for each action
ACTIONS_VECTORS = BASE_DIR + 'actions_vectors.json'
# File with the activities ordered
ACTIVITIES_ORDERED = BASE_DIR + 'activities.json'
# Dataset with vectors but without the action timestamps
DATASET_NO_TIME = BASE_DIR + 'dataset_no_time.json'
# Dataset with vectors but without the action timestamps, 2 channels
DATASET_NO_TIME_2_CHANNELS = BASE_DIR + 'dataset_no_time_2_channels.json'

# When there is no activity
NONE = 'None'
# Separator for the text file
SEP = ' '
# Maximun number of actions in an activity
ACTIVITY_MAX_LENGHT = 32
# word2vec dimensions for an action
ACTION_MAX_LENGHT = 50

DATASET = DATASET_KASTEREN_REDUCED

if DATASET == DATASET_KASTEREN_REDUCED:
    DELIMITER = ' '
elif DATASET == DATASET_KASTEREN:
    DELIMITER = ','

# Generates the text file from the csv
def process_csv(none=False):
    actions = ''    
    activities_set = set()
    actions_set = set()
    with open(DATASET, 'rb') as csvfile: # This only works with DATASET_KASTEREN now
        print 'Processing:', DATASET
        reader = csv.reader(csvfile, delimiter=DELIMITER)        
        i = 0
        for row in reader:
            i += 1
            if i == 1:
                continue        
            action = row[1]
            activity = row[2]
            if none:
                actions += action + SEP
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

# Generates a CSV dataset with periods
def transform_csv_to_periods():
    with open(DATASET, 'rb') as csvfile:
        with open(DATASET_ACTION_PERIODS, 'wb') as new_dataset:
              print 'Processing:', DATASET
              reader = csv.reader(csvfile, delimiter=DELIMITER)   
              writer = csv.writer(new_dataset, delimiter=DELIMITER)
              i = 0
              for row in reader:
                i += 1
                date = row[0]
                instant = row[1]
                sensor = row[2]
                action = row[3]
                event = row[4]
                period = instant_to_period(instant, 30)
                activity = row[5]
                action_period = action + '_' + str(period)
                new_row = [date, instant, sensor, action_period, event, activity]
                writer.writerow(new_row)
    print 'Total actions processed:', i
    
# Generates the text file from the csv taking into account the time periods
def process_time_csv(none=True):
    transform_csv_to_periods()
    actions = ''    
    activities_set = set()
    actions_set = set()
    with open(DATASET_ACTION_PERIODS, 'rb') as csvfile:
        print 'Processing:', DATASET
        reader = csv.reader(csvfile, delimiter=DELIMITER)   
        i = 0
        for row in reader:
            i += 1 
            #date = row[0]
            #instant = row[1]
            #sensor = row[2]
            action = row[3]
            #event = row[4]
            activity = row[5]
            if none:
                actions += action
                activities_set.add(activity)      
                actions_set.add(action)
            if activity != NONE and not none:
                actions += action
                activities_set.add(activity)
                actions_set.add(action)
            if i % 10000 == 0:
                print '  -Actions processed:', i
        print 'Total actions processed:', i
    
    with open(ACTION_TIME_TEXT, 'w') as textfile: 
        textfile.write(actions)     
    json.dump(list(activities_set), open(UNIQUE_ACTIVITIES, 'w'))
    json.dump(list(actions_set), open(UNIQUE_TIME_ACTIONS, 'w'))
    print 'Text file saved'
    
# Transforms a time instant (e.g. 09:33:41) to a period
def instant_to_period(instant, period_mins=30):
    hours = int(instant.split(':')[0])
    mins = int(instant.split(':')[1])
    current_minutes = ((hours * 60.0) + mins) 
    current_period = int(round(current_minutes/period_mins))
    return current_period
            

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
    process_time_csv(none=True)
#    create_vector_file()
#    order_activities(True)
#    calculate_statistics()
#    create_vector_dataset_no_time()
#    create_vector_dataset_no_time_2_channels()
    print 'Fin'



            


