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

