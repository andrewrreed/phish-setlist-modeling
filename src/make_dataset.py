#!/usr/bin/python
'''
Build MVP Modeling Data

The following script can be used to recreate the exact data used in MVP modeling.
To recreate the data, run the the following command from the terminal with the 
current working directory as 'phish-setlist-modeling/src':

python make_dataset.py

'''

import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from util import load_pickle_object, create_pickle_object, preprocess_data, create_song_encodings, encode_setlist_data, create_sequence_modeling_data

# load raw setlist data
all_setlists = load_pickle_object(r'../data/raw/extract-05032019/all_setlists.pickle')

# process and clean all setlists
setlist_list = preprocess_data(all_setlists)

# build song encodings
song_to_idx, idx_to_song = create_song_encodings(setlist_list)

# encode the full setlist
encoded_setlist_list = encode_setlist_data(song_to_idx, setlist_list)

# create X and y sequence data
X_data, y_data = create_sequence_modeling_data(encoded_setlist_list)

# create test/train split (Note - unable to stratify because some songs only occur once)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=2)

# pickle training data
create_pickle_object(obj= X_train, pickle_name='X_train.pickle', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= X_test, pickle_name='X_test.pickle', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= y_train, pickle_name='y_train.pickle', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= y_test, pickle_name='y_test.pickle', file_path='../data/processed/mvp-setlist-modeling/')

# one-hot encode training data
num_classes = len(song_to_idx)

X_train_hot = np.array([to_categorical(x, num_classes=num_classes) for x in X_train])
X_test_hot = np.array([to_categorical(x, num_classes=num_classes) for x in X_test])
y_train_hot = to_categorical(y_train, num_classes=num_classes)
y_test_hot = to_categorical(y_test, num_classes=num_classes)

# pickle one-hot encoded training data
create_pickle_object(obj= X_train_hot, pickle_name='X_train_hot.pickle', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= X_test_hot, pickle_name='X_test_hot.pickle', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= y_train_hot, pickle_name='y_train_hot.pickle', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= y_test_hot, pickle_name='y_test_hot.pickle', file_path='../data/processed/mvp-setlist-modeling/')
