import json
import sys
import math

from gensim.models import Word2Vec

import h5py

from keras.callbacks import ModelCheckpoint
from keras.layers import Dot, Bidirectional, Attention, Average, Concatenate, Convolution2D, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, MaxPooling2D, Multiply, Reshape
from keras.models import load_model, Model
from keras.preprocessing.text import Tokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# Kasteren dataset
DIR = '/sensor2vec/kasteren_dataset/'
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
BEST_MODEL = '/results/best_model.hdf5'

# if time is being taken into account
TIME = False

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
        
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if save == True:
            plt.savefig('/results/accuracy.png')
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
            plt.savefig('/results/loss.png')
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
    print(('total actions', len(actions)))
    print(('total timestaps', len(timestamps)))
    print((timestamps[0]))
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
        time_x, time_y = transform_time_cyclic(timestamp, False)
        time_coord =  [time_x, time_y]
        hours.append(time_coord)

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
    for action, i in list(action_index.items()):
        try:            
            embedding_vector = model[action]
            embedding_matrix[i] = embedding_vector            
        except:
            if action in unknown_words:
                unknown_words[action] += 1
            else:
                unknown_words[action] = 1
    print(("Number of unknown tokens: " + str(len(unknown_words))))
    print(unknown_words)
    
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
    print(('*' * 20))
    print('Loading dataset...')
    sys.stdout.flush()
    #dataset of activities
    DATASET = DATASET_CSV
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]    
    # we only need the actions without the period to calculate the onehot vector for y, because we are only predicting the actions
    unique_actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    total_actions = len(unique_actions)
    
    print(('*' * 20))
    print('Preparing dataset...')
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
    print(('Different actions:', total_actions))
    print(('Total examples:', total_examples))
    print(('Train examples:', len(X_actions_train), len(y_train))) 
    print(('Test examples:', len(X_actions_test), len(y_test)))
    sys.stdout.flush()  
    X_actions_train = np.array(X_actions_train)
    X_times_train = np.array(X_times_train)
    y_train = np.array(y_train)
    X_actions_test = np.array(X_actions_test)
    X_times_test = np.array(X_times_test)
    y_test = np.array(y_test)
    print('Shape (X,y):')
    print((X_actions_train.shape))
    print((X_times_train.shape))
    print((y_train.shape))
   
    print(('*' * 20))
    print('Building model...')
    sys.stdout.flush()
    
    #actions branch
    input_actions = Input(shape=(INPUT_ACTIONS,), dtype='int32', name='input_actions')
    embedding_actions = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=INPUT_ACTIONS, trainable=True, name='embedding_actions')(input_actions)
    bidirectional_gru = Bidirectional(GRU(512, input_shape=(INPUT_ACTIONS, ACTION_EMBEDDING_LENGTH), name='bidirectional_gru'))(embedding_actions)
    
    # attention
    dense_att_1 = Dense(512, activation = 'tanh',name = 'dense_att_1')(bidirectional_gru)
    dense_att_2 = Dense(INPUT_ACTIONS, activation = 'softmax',name = 'dense_att_2')(dense_att_1)
    reshape_att = Reshape((INPUT_ACTIONS, 1), name = 'reshape_att')(dense_att_2) #so we can multiply it with embeddings
    #apply the attention
    apply_att = Multiply()([embedding_actions, reshape_att])

    #actions times branch
    input_time = Input(shape=(INPUT_ACTIONS,2), dtype='float32', name='input_time')

    #convolutions
    reshape = Reshape((INPUT_ACTIONS, ACTION_EMBEDDING_LENGTH, 1), name = 'reshape')(apply_att) #add channel dimension for the CNNs
    #branching convolutions
    ngram_2 = Convolution2D(200, (2, ACTION_EMBEDDING_LENGTH), padding='valid',activation='relu', name = 'conv_2')(reshape)
    maxpool_2 = MaxPooling2D(pool_size=(INPUT_ACTIONS-2+1,1), name = 'pooling_2')(ngram_2)
    ngram_3 = Convolution2D(200, (3, ACTION_EMBEDDING_LENGTH), padding='valid',activation='relu', name = 'conv_3')(reshape)
    maxpool_3 = MaxPooling2D(pool_size=(INPUT_ACTIONS-3+1,1), name = 'pooling_3')(ngram_3)
    ngram_4 = Convolution2D(200, (4, ACTION_EMBEDDING_LENGTH), padding='valid',activation='relu', name = 'conv_4')(reshape)
    maxpool_4 = MaxPooling2D(pool_size=(INPUT_ACTIONS-4+1,1), name = 'pooling_4')(ngram_4)
    ngram_5 = Convolution2D(200, (5, ACTION_EMBEDDING_LENGTH), padding='valid',activation='relu', name = 'conv_5')(reshape)
    maxpool_5 = MaxPooling2D(pool_size=(INPUT_ACTIONS-5+1,1), name = 'pooling_5')(ngram_5)
    #1 branch again
    merged = Concatenate(axis=2)([maxpool_2, maxpool_3, maxpool_4, maxpool_5])
    flatten1 = Flatten(name = 'flatten1')(merged)
    flatten2 = Flatten(name = 'flatten2')(input_time)
    concat = Concatenate()([flatten1, flatten2])
    dense_1 = Dense(256, activation = 'relu',name = 'dense_1')(concat)
    drop_1 = Dropout(0.8, name = 'drop_1')(dense_1)
    #action prediction
    output_actions = Dense(total_actions, activation='softmax', name='main_output')(drop_1)

    model = Model([input_actions, input_time], [output_actions])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    print((model.summary()))
    sys.stdout.flush()
    
    print(('*' * 20))
    print('Training model...')    
    sys.stdout.flush()
    BATCH_SIZE = 128
    checkpoint = ModelCheckpoint(BEST_MODEL, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    history = model.fit([X_actions_train, X_times_train], y_train, batch_size=BATCH_SIZE, epochs=1000, validation_data=([X_actions_test, X_times_test], y_test), shuffle=False, callbacks=[checkpoint])

    print(('*' * 20))
    print('Plotting history...')
    sys.stdout.flush()
    plot_training_info(['accuracy', 'loss'], True, history.history)
    
    print(('*' * 20))
    print('Evaluating best model...')
    sys.stdout.flush()    
    model = load_model(BEST_MODEL)
    metrics = model.evaluate([X_actions_test, X_times_test], y_test, batch_size=BATCH_SIZE)
    print(metrics)
    
    predictions = model.predict([X_actions_test, X_times_test], BATCH_SIZE)
    correct = [0] * 5
    prediction_range = 5
    for i, prediction in enumerate(predictions):
        correct_answer = y_test[i].tolist().index(1)       
        best_n = np.sort(prediction)[::-1][:prediction_range]
        for j in range(prediction_range):
            if prediction.tolist().index(best_n[j]) == correct_answer:
                for k in range(j,prediction_range):
                    correct[k] += 1 
    
    accuracies = []                   
    for i in range(prediction_range):
        print(('%s prediction accuracy: %s' % (i+1, (correct[i] * 1.0) / len(y_test))))
        accuracies.append((correct[i] * 1.0) / len(y_test))
    
    print(accuracies)
    
    print(('************ FIN ************\n' * 3))  

if __name__ == "__main__":
    main(sys.argv)
