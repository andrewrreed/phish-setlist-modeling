#!/usr/bin/python
'''
Build MVP Modeling Data

The following script can be used to recreate the exact data used in MVP modeling.
To recreate the data, run the the following command from the terminal with the 
current working directory as 'phish-setlist-modeling/scriptsw':

python make_dataset.py

'''
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# check/add module to path
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.util import load_pickle_object, create_pickle_object
from src.process import preprocess_data, create_song_encodings, encode_setlist_data, create_sequence_modeling_data


# load raw setlist data
all_setlists = load_pickle_object(r'../data/raw/extract-05032019/all_setlists.pkl')

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

# pickle the song encoding mappings
create_pickle_object(obj= song_to_idx, pickle_name='song_to_idx.pkl', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= idx_to_song, pickle_name='idx_to_song.pkl', file_path='../data/processed/mvp-setlist-modeling/')

# pickle training data
create_pickle_object(obj= X_train, pickle_name='X_train.pkl', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= X_test, pickle_name='X_test.pkl', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= y_train, pickle_name='y_train.pkl', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= y_test, pickle_name='y_test.pkl', file_path='../data/processed/mvp-setlist-modeling/')

# one-hot encode training data
num_classes = len(song_to_idx)
X_train_hot = np.array([to_categorical(x, num_classes=num_classes) for x in X_train])
X_test_hot = np.array([to_categorical(x, num_classes=num_classes) for x in X_test])
y_train_hot = to_categorical(y_train, num_classes=num_classes)
y_test_hot = to_categorical(y_test, num_classes=num_classes)

# pickle one-hot encoded training data
## NOTE - one-hot encoded X datasets are too large to pickle AND are not needed for modeling
#create_pickle_object(obj= X_train_hot, pickle_name='X_train_hot.pkl', file_path='../data/processed/mvp-setlist-modeling/')
#create_pickle_object(obj= X_test_hot, pickle_name='X_test_hot.pkl', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= y_train_hot, pickle_name='y_train_hot.pkl', file_path='../data/processed/mvp-setlist-modeling/')
create_pickle_object(obj= y_test_hot, pickle_name='y_test_hot.pkl', file_path='../data/processed/mvp-setlist-modeling/')