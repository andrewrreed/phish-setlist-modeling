'''
Helper functions for deep learning modeling using Keras and Tensorflow

'''
# set seeds
from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(2)

# import 
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
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
    
    X_train = load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling/seqlen-{seq_len}/X_train.pkl')
    X_test = load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling/seqlen-{seq_len}/X_test.pkl')
    y_train_hot = load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling/seqlen-{seq_len}/y_train_hot.pkl')
    y_test_hot = load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling/seqlen-{seq_len}/y_test_hot.pkl')
    idx_to_song = load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling/seqlen-{seq_len}/idx_to_song.pkl')
    
    return X_train, X_test, y_train_hot, y_test_hot, idx_to_song


def nn_model(nn_arch, X_train, y_train, X_test, y_test, epochs, batch_size):
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
    
    Returns:
        model_history (keras.callbacks.History) - object storing relevant model history


    '''
    
    # extract model name
    NAME = nn_arch.name

    # define callbacks for Tensorboard logs and ModelCheckpoints
    callbacks = [TensorBoard(log_dir = f'../logs/{NAME}',
                            histogram_freq=1,
                            embeddings_freq=0,
                            embeddings_data=X_train
                            ),
                 ModelCheckpoint(
                            filepath=f'../models/mvp-setlist-modeling/model.{NAME}.hdf5',
                            monitor='val_acc',
                            save_best_only=True,
                            mode='max',
                            verbose=1
                            )]
    
    # compile model
    nn_arch.compile(optimizer='adam',
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
    
    return model_history

def train_model(nn_arch, nn_arch_params, epochs, batch_size):
    '''
    Function to train a predefined model architecture with varying model parameters including sequence length
    
    Args:
        nn_arch (str) - name of architecture to use from model.py (Ex: 'nn_arch_1')
        nn_arch_params (dict) - all parameters required for the given architecture type
        epochs - number of epochs to train for
        batch-size - size of batches to be computed in each forward/backward propogation
    
    Returns:
        model_history (keras.callbacks.History) - object storing relevant model history
    
    '''
    
        
    # load the required data based on sequence length
    seq_len = nn_arch_params['seq_length']
    X_train, X_test, y_train_hot, y_test_hot, idx_to_song = load_training_data(seq_len)
    
    # ensure number of classes from data is correct and add to nn_arch_params
    assert nn_arch_params['num_classes'] == len(idx_to_song), "Number of anticipated classes was incorrect!"
    
    # build keras model object from desired architecture
    if nn_arch == 'nn_arch_1':
        nn_arch_obj = nn_arch_1(**nn_arch_params)
    elif nn_arch == 'nn_arch_2':
        nn_arch_obj = nn_arch_2(**nn_arch_params)
    else:
        raise ValueError('Must enter a valid architecture function from model.py')
    
    
    # train the model based on data + architecture
    model = nn_model(nn_arch=nn_arch_obj,
                        X_train=X_train,
                        y_train=y_train_hot,
                        X_test=X_test,
                        y_test=y_test_hot,
                        epochs=epochs,
                        batch_size=batch_size)
    
    return model

# ------------------------- Architecture Types -------------------------

def nn_arch_1(seq_length, num_classes, lstm_units):
    '''
    Phish Setlist Modeling: Achitecture 1 (Baseline)
    
    A baseline, Recurrent Neural Network model consisting of:
        Embedding Layer - to create a vector space representation of each song Phish has played
        LSTM Layer - this recurrent layer allows the network to learn sequential patterns over time (variable number of units)
        Fully Connected Layer - a layer to digest the LSTM output
        Softmax Output - an output layer that represents one unit for each song, creating a multiclass classfication task
        
    Args:
        seq_length (int) - the input sequence lengths being fed to the model
        num_classes (int) - the number of unique songs to be learned in the embedding layer
        lstm_units (int) - number of units in the LSTM layer
    
    Returns:
        model (keras.engine) - a compiled keras model
    
    '''
    
    base_name = 'nn_arch_1'
    
    model = Sequential()
    model.add(Embedding(input_dim=num_classes, output_dim=50, input_length=seq_length, name='embed'))
    model.add(LSTM(units=lstm_units))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    model.name = f'{base_name}-{seq_length}-seqlen-{lstm_units}-lstmunits'
    
    return model
    
def nn_arch_2(seq_length, num_classes, lstm_units, dropout_before, dropout_after):
    '''
    Phish Setlist Modeling: Achitecture 2 (Dropout After)
    
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
    
    base_name = 'nn_arch_2'
    
    model = Sequential()
    model.add(Embedding(input_dim=num_classes, output_dim=50, input_length=seq_length, name='embed'))
    model.add(Dropout(rate=dropout_before, seed=2))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(rate=dropout_after, seed=2))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    model.name = f'{base_name}-{seq_length}-seqlen-{lstm_units}-lstmunits-{dropout_before}-b_dropout-{dropout_after}-a_dropout'
    
    return model