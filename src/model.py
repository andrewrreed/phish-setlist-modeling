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
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import TensorBoard, ModelCheckpoint



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
    
    X_train = src.util.load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling/seqlen-{seq_len}/X_train.pkl')
    X_test = src.util.load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling/seqlen-{seq_len}/X_test.pkl')
    y_train_hot = src.util.load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling/seqlen-{seq_len}/y_train_hot.pkl')
    y_test_hot = src.util.load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling/seqlen-{seq_len}/y_test_hot.pkl')
    idx_to_song = src.util.load_pickle_object(file_path=f'../data/processed/mvp-setlist-modeling/seqlen-{seq_len}/idx_to_song.pkl')
    
    return X_train, X_test, y_train_hot, y_test_hot, idx_to_song


def nn_model(nn_arch, X_train, y_train, X_test, y_test, epochs, batch_size):
    '''
    Function to train a Keras neural network model provided a compiled architecture and data inputs.
    
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

def nn_arch_1(seq_length, num_classes, lstm_units):
    '''
    Phish Setlist Modeling: Achitecture 1
    
    A baseline, Recurrent Neural Network model consisting of:
        Embedding Layer - to create a vector space representation of each song Phish has played
        LSTM Layer - this recurrent layer allows the network to learn sequential patterns over time (variable number of units)
        Fully Connected Layer - a layer to digest the LSTM output (variable dropout)
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
    model.add(Dense(units=lstm_units, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    model.name = f'{base_name}-{seq_length}-seqlen-{lstm_units}-lstmunits'
    
    return model
    