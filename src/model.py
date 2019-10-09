'''
Helper functions for deep learning modeling using Keras and Tensorflow

'''
# set seeds
from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(2)

# import 
import keras
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from src.util import load_pickle_object



def load_training_data(seq_len):
    '''
    Loads in all data needed for training a setlist prediction model.
    
    NOTE - Assumes all training data has be build using make_dataset.py located in src/scripts.
    
    Args:
        seq_len (int) - sequence length used to build the dataset in make_datset.py, used for folder routing
    Returns:
        X_train (ndarray) - 2D array representing X training data of shape = number of sequences by length of sequences
        X_test (ndarray) - 2D array representing X testing data of shape = number of sequences by length of sequences
        y_train_hot (ndarray) - 2D array representing one-hot encoded y training data for each sequence in X
        y_test_hot (ndarray) - 2D array representing one-hot encoded y testing data for each sequence in X
        idx_to_song (dict) - a mapping of numeric encodings to song titles
    
    '''
    
    X_train = load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling2/seqlen-{seq_len}/X_train.pkl')
    X_test = load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling2/seqlen-{seq_len}/X_test.pkl')
    y_train_hot = load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling2/seqlen-{seq_len}/y_train_hot.pkl')
    y_test_hot = load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling2/seqlen-{seq_len}/y_test_hot.pkl')
    idx_to_song = load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling2/seqlen-{seq_len}/idx_to_song.pkl')
    
    return X_train, X_test, y_train_hot, y_test_hot, idx_to_song


def train_model(nn_arch_type, nn_arch_params, epochs, batch_size, lr_finder=False):
    '''
    Function to train a predefined model architecture with varying model parameters including sequence length
    
    Args:
        nn_arch_type (str) - name of architecture to use from model.py (Ex: 'nn_arch_1')
        nn_arch_params (dict) - all parameters required for the given architecture type
        epochs (int) - number of epochs to train for
        batch-size (int) - size of batches to be computed in each forward/backward propogation
        lr_finder (bool) - toggle learning rate finder callback
    
    Returns:
        model_history (keras.callbacks.History) - object storing relevant model history
    
    '''
    
    # load the required data based on sequence length
    seq_len = nn_arch_params['seq_length']
    X_train, X_test, y_train_hot, y_test_hot, idx_to_song = load_training_data(seq_len)
    
    # ensure number of classes from data is correct and add to nn_arch_params
    assert nn_arch_params['num_classes'] == len(idx_to_song), "Number of anticipated classes was incorrect!"
    
    # build keras model object from desired architecture
    if nn_arch_type == 'nn_arch_1':
        nn_arch_obj = nn_arch_1(**nn_arch_params)
    elif nn_arch_type == 'nn_arch_2':
        nn_arch_obj = nn_arch_2(**nn_arch_params)
    elif nn_arch_type == 'nn_arch_3':
        nn_arch_obj = nn_arch_3(**nn_arch_params)
    else:
        raise ValueError('Must enter a valid architecture function from model.py')
    
    
    # train the model based on data + architecture
    model, lrn_finder = nn_model(nn_arch=nn_arch_obj,
                                X_train=X_train,
                                y_train=y_train_hot,
                                X_test=X_test,
                                y_test=y_test_hot,
                                epochs=epochs,
                                batch_size=batch_size,
                                lr_finder=lr_finder)
    if lr_finder == False:
        return model
    else:
        return model, lrn_finder

def nn_model(nn_arch, X_train, y_train, X_test, y_test, epochs, batch_size, lr_finder):
    '''
    Function to train a Keras neural network model provided a compiled architecture and traaining data inputs.
    
    Args:
        nn_arch (keras.engine) - a compiled keras model
        X_train - X features for training
        y_train - y features for training
        X_test - X features for evaluation
        y_test - y features for evaluation
        epochs - number of epochs to train for
        batch-size - size of batches to be computed in each forward/backward propogation
        lr_finder (bool) - toggle learning rate finder callback
    
    Returns:
        model_history (keras.callbacks.History) - object storing relevant model history


    '''
    
    # extract model name
    NAME = nn_arch.name

    # define callbacks for Tensorboard logs and ModelCheckpoints
    tensorboard = TensorBoard(log_dir = f'../logs2/{NAME}4',
                            histogram_freq=1,
                            embeddings_freq=0,
                            embeddings_data=X_train
                            )
    checkpoint = ModelCheckpoint(
                            filepath=f'../models/mvp-setlist-modeling2/model.{NAME}4.hdf5',
                            monitor='val_acc',
                            save_best_only=True,
                            mode='max',
                            verbose=1
                            )
    lrn_finder = LRFinder(min_lr=0.00001,
                            max_lr=0.1,
                            steps_per_epoch=np.ceil(X_train.shape[0]/batch_size),
                            epochs=5)

    # build callbacks list
    callbacks = [tensorboard, checkpoint]
    # toggle lr_finder
    if lr_finder == True:
        callbacks.append(lrn_finder)
    
    # build optimizer object
    lr_obj = keras.optimizers.adam(lr=0.002)

    # compile model
    nn_arch.compile(optimizer=lr_obj,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # fit model
    with tf.device('/gpu:0'):
        model_history = nn_arch.fit(x=X_train,
                                    y=y_train,
                                    validation_data=(X_test, y_test),
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    callbacks=callbacks)

    # clear session and remove data vars
    keras.backend.clear_session()
    
    return model_history, lrn_finder

# ------------------------- Architecture Types -------------------------

# def nn_arch_1(seq_length, num_classes, lstm_units):
#     '''
#     Phish Setlist Modeling: Achitecture 1 (Baseline)
    
#     A baseline, Recurrent Neural Network model consisting of:
#         Embedding Layer - to create a vector space representation of each song Phish has played
#         LSTM Layer - this recurrent layer allows the network to learn sequential patterns over time (variable number of units)
#         Fully Connected Layer - a layer to digest the LSTM output
#         Softmax Output - an output layer that represents one unit for each song, creating a multiclass classfication task
        
#     Args:
#         seq_length (int) - the input sequence lengths being fed to the model
#         num_classes (int) - the number of unique songs to be learned in the embedding layer
#         lstm_units (int) - number of units in the LSTM layer
    
#     Returns:
#         model (keras.engine) - a compiled keras model
    
#     '''
    
#     base_name = 'nn_arch_1'
    
#     model = Sequential()
#     model.add(Embedding(input_dim=num_classes, output_dim=50, input_length=seq_length, name='embed'))
#     model.add(LSTM(units=lstm_units))
#     model.add(Dense(units=100, activation='relu'))
#     model.add(Dense(units=num_classes, activation='softmax'))
    
#     model.name = f'{base_name}-{seq_length}-seqlen-{lstm_units}-lstmunits'
    
#     return model
    
def nn_arch_1(seq_length, num_classes, lstm_units, dropout_before, dropout_after):
    '''
    Phish Setlist Modeling: Achitecture 1
    
    An improved Recurrent Neural Network model that introduces dropout after the LSTM layer:
        Embedding Layer - to create a vector space representation of each song Phish has played
        LSTM Layer - this recurrent layer allows the network to learn sequential patterns over time (variable number of units)
        Dropout Layer - this layer will help regularize our network (variable dropout)
        Fully Connected Layer - a layer to digest the LSTM output
        Softmax Output - an output layer that represents one unit for each song, creating a multiclass classfication task
        
    Args:
        seq_length (int) - the input sequence lengths being fed to the model
        num_classes (int) - the number of unique songs to be learned in the embedding layer
        lstm_units (int) - number of units in the LSTM layer
        dropout_before (float) - percent of inputs before LSTM to be dropped (set to zero)
        dropout_after (float) - percent of inputs after LSTM to be dropped (set to zero)
    
    Returns:
        model (keras.engine) - a compiled keras model
    
    '''
    
    base_name = 'nn_arch_1'
    
    model = Sequential()
    model.add(Embedding(input_dim=num_classes, output_dim=50, input_length=seq_length, name='embed'))
    model.add(Dropout(rate=dropout_before, seed=2))
    model.add(LSTM(units=lstm_units, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(rate=dropout_after, seed=2))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    model.name = f'{base_name}-{seq_length}-seqlen-{lstm_units}-lstmunits-{dropout_before}-b_dropout-{dropout_after}-a_dropout'
    
    return model

def nn_arch_2(seq_length, num_classes, lstm_units, dropout_before, dropout_after):
    '''
    Phish Setlist Modeling: Achitecture 2 (Two LSTMs)
    
    An improved Recurrent Neural Network model that introduces two LSTM layers with dropout:
        Embedding Layer - to create a vector space representation of each song Phish has played
        LSTM Layer - this recurrent layer allows the network to learn sequential patterns over time (variable number of units)
        LSTM Layer - this recurrent layer allows the network to learn sequential patterns over time (variable number of units)
        Dropout Layer - this layer will help regularize our network (variable dropout)
        Fully Connected Layer - a layer to digest the LSTM output
        Softmax Output - an output layer that represents one unit for each song, creating a multiclass classfication task
        
    Args:
        seq_length (int) - the input sequence lengths being fed to the model
        num_classes (int) - the number of unique songs to be learned in the embedding layer
        lstm_units (int) - number of units in the LSTM layer
        dropout_before (float) - percent of inputs before LSTM to be dropped (set to zero)
        dropout_after (float) - percent of inputs after LSTM to be dropped (set to zero)
    
    Returns:
        model (keras.engine) - a compiled keras model
    
    '''
    
    base_name = 'nn_arch_2'
    
    model = Sequential()
    model.add(Embedding(input_dim=num_classes, output_dim=50, input_length=seq_length, name='embed'))
    model.add(Dropout(rate=dropout_before, seed=2))
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(Dropout(rate=dropout_after, seed=2))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(rate=dropout_after, seed=2))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    model.name = f'{base_name}-{seq_length}-seqlen-{lstm_units}-lstmunits-{dropout_before}-b_dropout-{dropout_after}-a_dropout'
    
    return model


# ------------------------- Learning Rate Finder -------------------------
# https://gist.github.com/jeremyjordan/ac0229abd4b2b7000aca1643e88e0f02

class LRFinder(Callback):
    
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''
    
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}
        
    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations 
        return self.min_lr + (self.max_lr-self.min_lr) * x
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        
    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        K.set_value(self.model.optimizer.lr, self.clr())
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()
    def plot_loss_smooth(self):
        '''Helper function to quickly observe the learning rate experiment results by smoothing curve.'''
        loss_smooth = savgol_filter(self.history['loss'], window_length=101, polyorder=2)
        plt.plot(self.history['lr'], loss_smooth)
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()

