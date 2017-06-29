# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:04:38 2016

@author: aitor
"""

import json
import random
import sys
import time

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import numpy as np

TRAINING_FILE = 'star_merged.txt'
text = open(TRAINING_FILE).read().lower()
print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
sys.stdout.flush()
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print 'SHAPE',X.shape

# build the model: 2 stacked LSTM
print('Build model...')
sys.stdout.flush()
DROP_OUT_VALUE = 0.4
NUMBER_OF_LSTMS = 2
model = Sequential()
for i in range(NUMBER_OF_LSTMS-1):
    model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
    model.add(Dropout(DROP_OUT_VALUE))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(DROP_OUT_VALUE))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    # if sum(pvals[:-1]) > 1 then random.multinomial does not work and it must 
    # be normalized.  
    if sum(a[:-1]) > 1:
#        print
#        print '*' * 10
#        print 'Normalizing'
#        print '-Before'
#        print '---sum(pvals[:-1]):', sum(a[:-1])
#        print '---Max e:', max(a[:-1])
        a = normalize(a)
#        print 'pvals'
#        print a
#        print 'pvals[:-1]', a[:-1]
#        print '-After'
#        print '---sum(pvals[:-1]):', sum(a[:-1])
#        print '---Max e:', max(a[:-1])
#    else:
#        print
#        print 'No normalization necessary'    
#        print '---sum(pvals[:-1]):', sum(a[:-1])
#        print '---Max e:', max(a[:-1])
    
    return np.argmax(np.random.multinomial(1, a, 1))
    
def normalize(pvals):
    total = sum(pvals[:-1])
    excess = np.float32(total - 1.0)
    excess = excess + np.finfo(np.float32).eps
    max_e = max(pvals[:-1])
    for i, elem in enumerate(pvals[:-1]):
        if elem == max_e:
#            print '  ---fixing'
#            print '  total:', total
#            print '  excess:', excess 
#            print i, elem
#            print '  staring value', pvals[:-1][i]
            new_value = pvals[:-1][i] - excess
#            print '  expected result',  new_value          
#            print '  - tipo del array:', type(pvals[:-1][i])
#            print '  - tipo del new value:', type(new_value)
#            print '  - funcion del setitem:', pvals[:-1][i].__setitem__
            pvals[:-1][i] = new_value
#            print '  - valor del array nuevo:', pvals[:-1][i]
#            print '  - es el mismo valor? ', pvals[:-1][i] == new_value
#            print '  - tipo del valor en el array:', type(pvals[:-1][i])
#            print '  - tipo del valor fuera del array:', type(new_value)
            break
    return pvals  
    
def save_model(model):
    json_string = model.to_json()
    model_name = 'model_%sLSTM_%s' % (NUMBER_OF_LSTMS, DROP_OUT_VALUE)
    open(model_name + '.json', 'w').write(json_string)
    model.save_weights(model_name + '.h5', overwrite=True)
    
def load_model(model_file, weights_file):
    model = model_from_json(open(model_file).read())
    model.load_weights(weights_file)

print ('Config:')
print ('Training file:', TRAINING_FILE)
print ('Number of LSTMs:', NUMBER_OF_LSTMS)
print ('Dropout:', DROP_OUT_VALUE)
total_history = {}
# train the model, output generated text after each iteration
for iteration in range(1, 160):
    if iteration % 20 == 0:
        print 'Saving model...'
        sys.stdout.flush()
        save_model(model)
        print 'Model saved'
        sys.stdout.flush()
    print()
    print('-' * 50)
    print('Iteration', iteration)
    sys.stdout.flush()
    print 'Start:', time.ctime()
    history = model.fit(X, y, validation_split=0.33, batch_size=128, nb_epoch=1)
    print 'End:', time.ctime()
    print history.history
    total_history[iteration] = history.history

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
        sys.stdout.flush()    

print 'Saving model...'
sys.stdout.flush()
json.dump(total_history, open('history.json', 'w'))
save_model(model)
print 'Model saved'
        