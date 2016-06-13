# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:24:51 2016

@author: aitor
"""

from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
import numpy as np

WINDOW_SIZE = 1
TRAIN_PER = 0.8
FEATURES = []

# Create the X and Y secuence vectors based on a window size
# E.G:
# training = [1,2,3,4,5], WINDOW_SIZE = 2 
# X --> [[1, 2], [2, 3], [3, 4]] 
# Y --> [3, 4, 5]
def create_X_Y(samples):
    X = []
    Y = []
    i = 0
    for i in range(len(samples)):
        if i + WINDOW_SIZE >= len(samples):
            break
#        X_sample = samples[i:i+WINDOW_SIZE]
        X_sample = samples[i]
        Y_sample = samples[i+WINDOW_SIZE]
        X.append(X_sample)
        Y.append(Y_sample)
    return X, Y
    
def encode_char(char):
    char_repr = [0] * len(FEATURES)
    pos = FEATURES.index(char)
    char_repr[pos] = 1
    return char_repr
    
def decode_char(char_repr):
    pos = char_repr.index(1)
    char = FEATURES[pos]
    return char
    
def get_most_probable(result):
    m = max(result)
    pos = [i for i, j in enumerate(result) if j == m]
    if len(pos) > 1:
        print 'Two probable values, picking only the first one'
    char_repr = [0] * len(FEATURES)
    char_repr[pos[0]] = 1
    return char_repr
    
    
def generate_text(model, number_of_chars = 2000, initial_char='\n'):
    generated_text = ''
    encoded_char = np.array([encode_char(initial_char)])
    for i in range(number_of_chars):
        result = model.predict(encoded_char, batch_size=1)
        generated_text += decode_char(result[0])
        encoded_char = result    

# *********Create the training and test data
print 'Creating test data...'
lines = []
for l in open('shakespear.txt', 'r'):
    lines.append(l)
text = ''.join(lines)
text = text.lower()
# Get the features (alphabet + other char)
chars = set()
for char in text:
    chars.add(char)
feature_list = list(chars)
FEATURES = feature_list
# Transform the chars to feature_list vectors. shape = [len(feature_list)] with
# a 1 on the pos of the char in feature_list
vectorized_text = []
for char in text:
    char_repr = encode_char(char)
    vectorized_text.append(char_repr)
# create the training and text X and Y groups. 
limit = int(TRAIN_PER * len(vectorized_text))
training = vectorized_text[:limit]
evaluation = vectorized_text[limit:]
X_train, Y_train = create_X_Y(training)
print 'Total samples training:', len(training)
print 'Total samples evaluation:', len(evaluation)
print 'Training sets lenght:', len(X_train), len(Y_train)
print 'Sample lenghts train:', set([len(x) for x in X_train]),set([len(x) for x in Y_train])
X_eval, Y_eval = create_X_Y(evaluation)
print 'Evaluation sets lenght:', len(X_eval), len(Y_eval)
print 'Sample lenghts train:', set([len(x) for x in X_eval]),set([len(x) for x in Y_eval])
X_train = np.array(X_train)
Y_train = np.array(Y_train)
#X_eval = np.array(X_eval)
#Y_eval = np.array(Y_eval)
print 'X_train shape:', X_train.shape
print 'Y_train shape:', Y_train.shape



# Create the model (WINDOW_SIZE, len(feature_list))
print 'Creating model...'
model = Sequential()
# This works
model.add(Embedding(X_train.shape[0], len(feature_list), input_length=len(feature_list)))
model.add(LSTM(len(feature_list))) 
model.add(Activation('softmax'))
print 'Training model...'
batch_size = 10
model.compile(optimizer='adam', loss='categorical_crossentropy')
print X_train.shape, Y_train.shape
model.fit(X_train, Y_train, nb_epoch=1, batch_size=batch_size)
print 'Model trained'
print 'Creating input data...'
char_repr = encode_char('a')
input_data = np.array([])
print 'Predicting...'
res = model.predict(input_data, batch_size=1)
print 'Prediction:', res
rep = get_most_probable(res[0])
print 'Char:', decode_char(rep)
#print 'Evaluating model...'
#score, acc = model.evaluate(X_eval, Y_eval, batch_size=batch_size)
#print('Test score:', score)
#print('Test accuracy:', acc)