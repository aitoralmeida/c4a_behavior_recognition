# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:24:51 2016

@author: aitor
"""

#from keras.models import Sequential
#from keras.layers.recurrent import LSTM, GRU
#from keras.layers.core import Dense, Dropout, Activation

WINDOW_SIZE = 10
TRAIN_PER = 0.8

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
        X_sample = samples[i:i+WINDOW_SIZE]
        Y_sample = samples[i+WINDOW_SIZE]
        X.append(X_sample)
        Y.append(Y_sample)
    return X, Y



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
# Transform the chars to feature_list vectors. shape = [len(feature_list)] with
# a 1 on the pos of the char in feature_list
vectorized_text = []
for char in text:
    char_repr = [0] * len(feature_list)
    pos = feature_list.index(char)
    char_repr[pos] = 1
    vectorized_text.append(char_repr)
# create the training and text X and Y groups. 
limit = int(TRAIN_PER * len(vectorized_text))
training = vectorized_text[:limit]
evaluation = vectorized_text[limit:]
X_train, Y_train = create_X_Y(training)
X_eval, Y_eval = create_X_Y(evaluation)

## Create the model
#model = Sequential()
#model.add(LSTM(32, input_shape=(WINDOW_SIZE, len(feature_list))))
#model.add(Activation('softmax'))
#
#model.compile(optimizer='adam', loss='categorical_crossentropy')