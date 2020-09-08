import json
import sys
import h5py

import tensorflow as tf

from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs, get_custom_objects
from keras_bert import Tokenizer as BertTokenizer
from keras_bert import layers as KerasBertLayers
from keras_bert import AdamWarmup, calc_train_steps

from keras_transformer import get_custom_objects as get_encoder_custom_objects

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dot, Bidirectional, Concatenate, Convolution2D, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, MaxPooling2D, Multiply, Reshape
from keras.models import load_model, Model
from keras.preprocessing.text import Tokenizer

from keras_pos_embd import PositionEmbedding

from tqdm import tqdm

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

#number of input actions for the model
INPUT_ACTIONS = 5
#Number of elements in the action's embbeding vector
ACTION_EMBEDDING_LENGTH = 50

#best model in the training
BEST_MODEL = '/results/best_model.hdf5'

# if time is being taken into account
TIME = False

BATCH_SIZE = 128

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
#    print actions.tolist()
#    print actions.tolist().index('HallBedroomDoor_1')
    # Use tokenizer to generate indices for every action
    # Very important to put lower=False, since the Word2Vec model
    # has the action names with some capital letters
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(actions.tolist())
    action_index = tokenizer.word_index  
#    print action_index
    #translate actions to indexes
    actions_by_index = []
    
    print((len(actions)))
    for action in actions:
#        print action
        actions_by_index.append(action_index[action])

    #Create the trainning sets of sequences with a lenght of INPUT_ACTIONS
    last_action = len(actions) - 1
    X = []
    y = []
    for i in range(last_action-INPUT_ACTIONS):
        X.append(actions_by_index[i:i+INPUT_ACTIONS])
        #represent the target action as a onehot for the softmax
        target_action = ''.join(i for i in actions[i+INPUT_ACTIONS] if not i.isdigit()) # remove the period if it exists
        y.append(unique_actions.index(target_action))
    return X, y, tokenizer   
    
"""
Prepares the training examples of secuences based on the total actions, using 
one hot vectors to represent them
Input
    df:Pandas DataFrame with timestamp, sensor, action, event and activity
    unique_actions: list of actions
Output:
    X: array with action index sequences
    y: array with action index for next action    
"""            
def prepare_x_y_onehot(df, unique_actions):
    #recover all the actions in order.
    actions = df['action'].values
    #translate actions to onehots
    actions_by_onehot = [] 
    for action in actions:
        onehot = [0] * len(unique_actions)
        action_index = unique_actions.index(action)
        onehot[action_index] = 1
        actions_by_onehot.append(onehot)

    #Create the trainning sets of sequences with a lenght of INPUT_ACTIONS
    last_action = len(actions) - 1
    X = []
    y = []
    for i in range(last_action-INPUT_ACTIONS):
        X.append(actions_by_onehot[i:i+INPUT_ACTIONS])
        #represent the target action as a onehot for the softmax
        target_action = actions_by_onehot[i+INPUT_ACTIONS]
        y.append(target_action)
    return X, y 

def main(argv):
    print(('*' * 20))
    print('Loading dataset...')
    sys.stdout.flush()
    #dataset of activities
    if TIME:
        DATASET = DATASET_ACTION_PERIODS
    else:
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
    X, y, tokenizer = prepare_x_y(df_dataset, unique_actions)    

    #divide the examples in training and validation
    total_examples = len(X)
    test_per = 0.2
    limit = int(test_per * total_examples)
    X_train = X[limit:]
    X_test = X[:limit]
    y_train = y[limit:]
    y_test = y[:limit]
    print(('Different actions:', total_actions))
    print(('Total examples:', total_examples))
    print(('Train examples:', len(X_train), len(y_train))) 
    print(('Test examples:', len(X_test), len(y_test)))
    sys.stdout.flush()  
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('Shape (X,y):')
    print((X_train.shape))
    print((y_train.shape))

    action_seq_pairs_train = []
    i = 0
    while i < len(X_train) - 1:
        action_seq_pairs_train.append([list(X_train[i]), list(X_train[i+1])])
        i += 2
    
    token_dict = get_base_dict()
    for pairs in action_seq_pairs_train:
        for token in pairs[0] + pairs[1]:
            if token not in token_dict:
                token_dict[token] = len(token_dict)

    print(token_dict)

    def transform_data(X_train, X_test, y_train, y_test):
        tokenizer = BertTokenizer(token_dict)

        indices_train, labels_train = [], []
        indices_test, labels_test = [], []

        for i in range(0, len(X_train)):
            ids, segments = tokenizer.encode(str(X_train[i]).replace("[", "").replace("]", ""), max_len=5)
            indices_train.append(ids)
            labels_train.append(y_train[i])
        items = list(zip(indices_train, labels_train))
        np.random.shuffle(items)
        indices_train, labels_train = zip(*items)
        indices_train = np.array(indices_train)
        mod = indices_train.shape[0] % BATCH_SIZE
        if mod > 0:
            indices_train, labels_train = indices_train[:-mod], labels_train[:-mod]
        
        for i in range(0, len(X_test)):
            ids, segments = tokenizer.encode(str(X_test[i]).replace("[", "").replace("]", ""), max_len=5)
            indices_test.append(ids)
            labels_test.append(y_test[i])
        items = list(zip(indices_test, labels_test))
        np.random.shuffle(items)
        indices_test, labels_test = zip(*items)
        indices_test = np.array(indices_test)
        mod = indices_test.shape[0] % BATCH_SIZE
        if mod > 0:
            indices_test, labels_test = indices_test[:-mod], labels_test[:-mod]
        
        return [indices_train, np.zeros_like(indices_train)], np.array(labels_train), [indices_test, np.zeros_like(indices_test)], np.array(labels_test)

    X_train, y_train, X_test, y_test = transform_data(X_train, X_test, y_train, y_test)
    
    executions = 100
    accuracies_avg = np.array([0, 0, 0, 0, 0])
    accuracies_best = np.array([0, 0, 0, 0, 0])

    for i in range(0, executions):
        
        print(('*' * 20))
        print('Building model...')
        sys.stdout.flush()

        model = get_model(
            token_num=len(token_dict),
            head_num=5,
            transformer_num=6,
            embed_dim=50,
            feed_forward_dim=50,
            seq_len=5,
            pos_num=20,
            dropout_rate=0.05,
        )
        # compile_model(model)
        model.summary()

        # def _generator():
        #     while True:
        #         yield gen_batch_inputs(
        #             action_seq_pairs_train,
        #             token_dict,
        #             token_list,
        #             seq_len=5,
        #             mask_rate=0.3,
        #             swap_sentence_rate=1.0,
        #         )

        # model.fit_generator(
        #     generator=_generator(),
        #     steps_per_epoch=1000,
        #     epochs=1,
        #     validation_data=_generator(),
        #     validation_steps=100,
        #     callbacks=[
        #         keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        #     ],
        # )

        # from keras_bert import extract_embeddings
        # def convert_list_of_int_to_str(lst):
        #     lst = [str(int) for i in lst]
        #     return ' '.join(lst)
        # actions_seqs = [convert_list_of_int_to_str([6, 6, 1, 1, 4]), convert_list_of_int_to_str([6, 6, 1, 1, 9])]
        # embeddings = extract_embeddings(new_model, actions_seqs, vocabs=token_dict)
        # print(embeddings)
        
        inputs = model.inputs[:2]

        dense = model.get_layer('NSP-Dense').output
        dense_1 = Dense(1024, activation = 'relu',name = 'dense_1')(dense)
        drop_1 = Dropout(0.8, name = 'drop_1')(dense_1)
        dense_2 = Dense(1024, activation = 'relu',name = 'dense_2')(drop_1)
        drop_2 = Dropout(0.8, name = 'drop_2')(dense_2)
        output_actions = Dense(total_actions, activation='softmax', name='main_output')(drop_2)

        train_x = np.random.standard_normal((1024, 100))

        total_steps, warmup_steps = calc_train_steps(
            num_example=train_x.shape[0],
            batch_size=32,
            epochs=10,
            warmup_proportion=0.1,
        )

        optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)

        model = Model(inputs, output_actions)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', 'mse', 'mae'])
        model.summary()
        
        print(('*' * 20))
        print('Training model...')    
        sys.stdout.flush()
        checkpoint = ModelCheckpoint(BEST_MODEL, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=50)
        history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=1000, validation_data=(X_test, y_test), shuffle=True, callbacks=[checkpoint, early_stopping])

        print(('*' * 20))
        print('Plotting history...')
        sys.stdout.flush()
        plot_training_info(['accuracy', 'loss'], True, history.history)
        
        print(('*' * 20))
        print('Evaluating best model...')
        sys.stdout.flush()
        custom_objects = get_encoder_custom_objects()
        custom_objects['TokenEmbedding'] = KerasBertLayers.TokenEmbedding
        custom_objects['PositionEmbedding'] = PositionEmbedding
        custom_objects['Extract'] = KerasBertLayers.Extract
        custom_objects['AdamWarmup'] = AdamWarmup
        model = load_model(BEST_MODEL, custom_objects=custom_objects)
        metrics = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
        print(metrics)
        
        predictions = model.predict(X_test, BATCH_SIZE)
        correct = [0] * 5
        prediction_range = 5
        for i, prediction in enumerate(predictions):
            correct_answer = y_test[i]     
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
        accuracies_best = np.max([accuracies_best, np.array(accuracies)], axis=0)
        accuracies_avg = np.array(accuracies) + accuracies_avg

        # https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
        tf.keras.backend.clear_session()

        print(('************ FIN ************\n' * 3))
    
    accuracies_avg = [x / executions for x in accuracies_avg]

    print(('************ AVG ************\n'))
    print(accuracies_avg)
    print(('************ BEST ************\n'))
    print(accuracies_best)

    print(('************ FIN MEDIA Y MEJOR RESULTADO ************\n' * 3))

if __name__ == "__main__":
    main(sys.argv)
