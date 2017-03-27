# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:12:22 2017

@author: aitor
"""
import json
import math
import sys

from gensim.models import Word2Vec

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, Embedding, Input, LSTM, merge, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, Model
from keras.preprocessing.text import Tokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# Kasteren dataset
DIR = './sensor2vec/kasteren_dataset/'
# Dataset with vectors but without the action timestamps
DATASET_CSV = DIR + 'base_kasteren_reduced.csv'
DATASET_NO_TIME = DIR + 'dataset_no_time.json'
# dataset with actions transformed with time periods
DATASET_ACTION_PERIODS = DIR + 'kasteren_action_periods.csv'
# List of unique activities in the dataset
UNIQUE_ACTIVITIES = DIR + 'unique_activities.json'
# List of unique actions in the dataset
UNIQUE_ACTIONS = DIR + 'unique_actions.json'
# List of unique actions in the dataset taking into account time periods
UNIQUE_TIME_ACTIONS = DIR + 'unique_time_actions.json'
# Action vectors
#ACTION_VECTORS = DIR + 'actions_vectors.json'
# Word2Vec model
WORD2VEC_MODEL = DIR + 'actions.model'
# Word2Vec model taking into account time periods
WORD2VEC_TIME_MODEL = DIR + 'actions_time.model'

#number of input actions for the model
INPUT_ACTIONS = 5
#Number of elements in the action's embbeding vector
ACTION_EMBEDDING_LENGTH = 50

#best model in the training
BEST_MODEL = 'best_model.hdf5'


"""
Load the best model saved in the checkpoint callback
"""
def select_best_model():
    model = load_model(BEST_MODEL)
    return model

"""
Function used to visualize the training history
metrics: Visualized metrics,
save: if the png are saved to disk
history: training history to be visualized
"""
def plot_training_info(metrics, save, history):
    # summarize history for accuracy
    if 'accuracy' in metrics:
        
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if save == True:
            plt.savefig('accuracy.png')
            plt.gcf().clear()
        else:
            plt.show()

    # summarize history for loss
    if 'loss' in metrics:
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        plt.legend(['train', 'test'], loc='upper left')
        if save == True:
            plt.savefig('loss.png')
            plt.gcf().clear()
        else:
            plt.show()
            
"""
Prepares the training examples of secuences based on the total actions, using
embeddings to represent them.
Input
    df:Pandas DataFrame with timestamp, sensor, action, event and activity
    unique_actions: list of actions
Output:
    X: array with action index sequences
    y: array with action index for next action
    tokenizer: instance of Tokenizer class used for action/index convertion
    
"""            
def prepare_x_y(df, unique_actions):
    #recover all the actions in order.
    actions = df['action'].values
    timestamps = df.index.tolist()
    print 'total actions', len(actions)
    print 'total timestaps', len(timestamps)
    print timestamps[0]
    # Use tokenizer to generate indices for every action
    # Very important to put lower=False, since the Word2Vec model
    # has the action names with some capital letters
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(actions.tolist())
    action_index = tokenizer.word_index  
    #translate actions to indexes
    actions_by_index = []
    for action in actions:
        actions_by_index.append(action_index[action])
        
    #translate timestamps to hours (format 2008-02-25 00:20:14)
    hours = []
    for timestamp in timestamps:
        hours.append(timestamp.hour)

    #Create the trainning sets of sequences with a lenght of INPUT_ACTIONS
    last_action = len(actions) - 1
    X_actions = []
    X_times = []
    y = []
    for i in range(last_action-INPUT_ACTIONS):
        X_actions.append(actions_by_index[i:i+INPUT_ACTIONS])
        X_times.append(hours[i:i+INPUT_ACTIONS])
        #represent the target action as a onehot for the softmax
        target_action = ''.join(i for i in actions[i+INPUT_ACTIONS] if not i.isdigit()) # remove the period if it exists
        target_action_onehot = np.zeros(len(unique_actions))
        target_action_onehot[unique_actions.index(target_action)] = 1.0
        y.append(target_action_onehot)
    return X_actions, X_times, y, tokenizer   
    
"""
Function to create the embedding matrix, which will be used to initialize
the embedding layer of the network
Input:
    tokenizer: instance of Tokenizer class used for action/index convertion
Output:
    embedding_matrix: matrix with the embedding vectors for each action
    
"""
def create_embedding_matrix(tokenizer):
    model = Word2Vec.load(WORD2VEC_MODEL)    
    action_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(action_index) + 1, ACTION_EMBEDDING_LENGTH))
    unknown_words = {}    
    for action, i in action_index.items():
        try:            
            embedding_vector = model[action]
            embedding_matrix[i] = embedding_vector            
        except:
            if action in unknown_words:
                unknown_words[action] += 1
            else:
                unknown_words[action] = 1
    print "Number of unknown tokens: " + str(len(unknown_words))
    print unknown_words
    
    return embedding_matrix
    
def transform_time_cyclic(timestamp, weekday):
    """
    This function transforms a timestamp into a cyclic clock-based time representation
    Parameters
    ----------        
    timestamp : datetime.datetime
        the timestamp to be transformed
    weekday : boolean
        a boolean to say whether the weekday should be treated for the calculation
                    
    Returns
    ----------
    x : float
        x coordinate of the 2D plane defining the clock [-1, 1]
    y : float
        y coordinate of the 2D plane defining the clock [-1, 1]
    """
    # Timestamp comes in datetime.datetime format
    HOURS = 24
    MINUTES = 60
    SECONDS = 60
    
    MAX_SECONDS = 0.0
    total_seconds = -1.0 # For error checking
    
    if weekday == True:    
        MAX_SECONDS = float(6*HOURS*MINUTES*SECONDS + 23*MINUTES*SECONDS + 59*SECONDS + 59)
        total_seconds = float(timestamp.weekday()*HOURS*MINUTES*SECONDS + timestamp.hour*MINUTES*SECONDS + timestamp.minute*SECONDS + timestamp.second)
    else:
        MAX_SECONDS = float(23*MINUTES*SECONDS + 59*SECONDS + 59)
        total_seconds = float(timestamp.hour*MINUTES*SECONDS + timestamp.minute*SECONDS + timestamp.second)
    
        
    angle = (total_seconds*2*math.pi) / MAX_SECONDS
    
    x = math.cos(angle)
    y = math.sin(angle)
    
    return x, y


def main(argv):
    print '*' * 20
    print 'Loading dataset...'
    sys.stdout.flush()
    #dataset of activities
    DATASET = DATASET_CSV
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]    
    # we only need the actions without the period to calculate the onehot vector for y, because we are only predicting the actions
    unique_actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    total_actions = len(unique_actions)
    
    print '*' * 20
    print 'Preparing dataset...'
    sys.stdout.flush()
    # Prepare sequences using action indices
    # Each action will be an index which will point to an action vector
    # in the weights matrix of the Embedding layer of the network input
    X_actions, X_times, y, tokenizer = prepare_x_y(df_dataset, unique_actions)    
    # Create the embedding matrix for the embedding layer initialization
    embedding_matrix = create_embedding_matrix(tokenizer)
    
    #divide the examples in training and validation
    total_examples = len(X_actions)
    test_per = 0.2
    limit = int(test_per * total_examples)
    X_actions_train = X_actions[limit:]
    X_times_train = X_times[limit:]
    X_actions_test = X_actions[:limit]
    X_times_test = X_times[:limit]
    y_train = y[limit:]
    y_test = y[:limit]
    print 'Different actions:', total_actions
    print 'Total examples:', total_examples
    print 'Train examples:', len(X_actions_train), len(y_train) 
    print 'Test examples:', len(X_actions_test), len(y_test)
    sys.stdout.flush()  
    X_actions_train = np.array(X_actions_train)
    X_times_train = np.array(X_times_train)
    y_train = np.array(y_train)
    X_actions_test = np.array(X_actions_test)
    X_times_test = np.array(X_times_test)
    y_test = np.array(y_test)
    print 'Shape (X,y):'
    print X_actions_train.shape
    print X_times_train.shape
    print y_train.shape
   
    print '*' * 20
    print 'Building model...'
    sys.stdout.flush()
    # Actions embeddings branch
    input_actions = Input(shape=(INPUT_ACTIONS,), dtype='int32', name='input_actions')
    embedding_actions = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=INPUT_ACTIONS, trainable=True, name='embedding_actions')(input_actions)    
    # Actions times branch
    input_time = Input(shape=(INPUT_ACTIONS,), dtype='float32', name='input_time')
    reshape_1 = Reshape((INPUT_ACTIONS, 1))(input_time)
    #merge embeddings (5 x 50) and times (5 x 1), to have 5 x 51
    concat = merge([embedding_actions, reshape_1], mode='concat', concat_axis=-1)   
    # Everything continues in a single branch
    lstm_1 = LSTM(512, return_sequences=False, input_shape=(INPUT_ACTIONS, ACTION_EMBEDDING_LENGTH+1), name='lstm_1')(concat)
    dense_1 = Dense(1024, activation = 'relu',name = 'dense_1')(lstm_1)
    drop_1 = Dropout(0.8, name = 'drop_1')(dense_1)
    dense_2 = Dense(1024, activation = 'relu',name = 'dense_2')(drop_1)
    drop_2 = Dropout(0.8, name = 'drop_2')(dense_2)
    output_actions = Dense(total_actions, activation='softmax', name='main_output')(drop_2)
    
    model = Model(input=[input_actions, input_time], output=[output_actions])
        
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    print(model.summary())
    sys.stdout.flush()
    
    print '*' * 20
    print 'Training model...'    
    sys.stdout.flush()
    BATCH_SIZE = 128
    checkpoint = ModelCheckpoint(BEST_MODEL, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    history = model.fit([X_actions_train, X_times_train], y_train, batch_size=BATCH_SIZE, nb_epoch=1000, validation_data=([X_actions_test, X_times_test], y_test), shuffle=False, callbacks=[checkpoint])

    print '*' * 20
    print 'Plotting history...'
    sys.stdout.flush()
    plot_training_info(['accuracy', 'loss'], True, history.history)
    

    print '*' * 20
    print 'Evaluating best model...'
    sys.stdout.flush()    
    model = load_model(BEST_MODEL)
    metrics = model.evaluate([X_actions_test, X_times_test], y_test, batch_size=BATCH_SIZE)
    print metrics
    
    print '************ FIN ************\n' * 3  


if __name__ == "__main__":
    main(sys.argv)
