# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:12:22 2017

@author: aitor
"""
from gensim.models import Word2Vec

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np


"""
Function used to save a model to disk.
model: model to be saved.
"""
def save_model(model):
    json_string = model.to_json()
    model_name = 'model_activity_lstm'
    open(model_name + '.json', 'w').write(json_string)
    model.save_weights(model_name + '.h5', overwrite=True)

"""
Function used to load a model previously saved to disk.
model_file: configuration of the model layers.
weights_file: value of the model weights.
"""    
def load_model(model_file, weights_file):
    model = model_from_json(open(model_file).read())
    model.load_weights(weights_file)

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

def main(argv):
    print '*' * 20
    print 'Building model...'
    sys.stdout.flush()
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=ACTIVITY_MAX_LENGHT, trainable=True))
    model.add(LSTM(512, return_sequences=False, dropout_W=0.2, dropout_U=0.2, input_shape=(ACTIVITY_MAX_LENGHT, ACTION_MAX_LENGHT)))  
    model.add(Dense(total_activities))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    print(model.summary())
    sys.stdout.flush()
    
    print '*' * 20
    print 'Training model...'    
    sys.stdout.flush()
    history = model.fit(X, y, batch_size=16, nb_epoch=1000, validation_data=(X_test, y_test), shuffle=False)
    
    print '*' * 20    
    print 'Saving model...'
    sys.stdout.flush()
    save_model(model)
    print 'Plotting history...'
    plot_training_info(['accuracy', 'loss'], True, history.history)
    print '************ FIN ************'    

if __name__ == "__main__":
main(sys.argv)
