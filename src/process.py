'''
Data processing functions for the phish-setlist-modeling package

'''
import os
import pickle
import numpy as np
import pandas as pd
import random
from bs4 import BeautifulSoup

# ------------------------- Processing Utilities -------------------------

def preprocess_data(all_setlists_df):
    '''
    Function to ingest, clean, and process a dataframe of Phish setlists into a "corpus"-like
    list songs and set identifiers

    Args:
        all_setlists_df (dataframe) - a dataframe as returned by the "get_all_setlists()" method in the pyphishnet API wrapper (link below)

    Returns:
        setlist_list (list) - a list to serve as a "corpus" representation of all Phish setlists in chronological order, including identifiers

    NOTE - the pyphishnet API wrapper is accessible here: https://github.com/areed1242/pyphishnet.git 

    '''
    
    ## ------- Cleanse setlists -------
    
    # filter to only Phish setlist data
    all_setlists = all_setlists_df[all_setlists_df.artistid == 1]
    # reset index
    all_setlists.reset_index(drop=True, inplace=True)
    
    # create a new dataframe that has ONLY complete datasets (i.e. has Set 1, Set 2, and Encore)
    complete_setlists = pd.DataFrame()

    for i, row in all_setlists.iterrows():
        # get setlist as list
        setlist = row.setlistdata_clean.split('|')
        # Check for presence of Set 1, Set 2, and Encore
        if 'Set 1' and 'Set 2' and 'Encore' in setlist:
            complete_setlists = complete_setlists.append(row)
    
    # reset index
    complete_setlists.reset_index(drop=True, inplace=True)
    
    print(f'{complete_setlists.shape[0]} of the {all_setlists.shape[0]} setlists have a Set 1, Set 2, and an Encore section')
    print()
    
    
    ## ------- Build a full "corpus" of songs -------

    setlist_list = []

    for i, row in complete_setlists.iterrows():

        # add a ', ' unless its the last record
        if i == complete_setlists.shape[0]-1:
            setlist = row.setlistdata_clean
        else:
            setlist = row.setlistdata_clean + '|'

        # append to full list
        setlist_list.append(setlist)

    # join to one long string
    setlist_string = ''.join(setlist_list)
    
    
    ## ------- Create unique identifiers for sets/encores -------
    
    # replace ids
    setlist_string = setlist_string.replace('Set 1', '<SET1>').replace('Set 2', '<SET2>').replace('Set 3', '<SET3>').replace('Set 4', '<SET4>').replace('Encore 2', '<ENCORE2>').replace('Encore', '<ENCORE>')
    
    # split string data into list
    setlist_list = setlist_string.split('|')

    return setlist_list


def create_song_encodings(setlist_list):
    '''
    Creates a numeric encoding for each song in the input string, as well as a reverse mapping for easy
    song lookups

    Args:
        setlist_list (list) - the object returned from the preprocess_data utility function

    Returns:
        song_to_idx (dict) - a mapping of song titles to numeric encodings
        idx_to_song (dict) - a mapping of numeric encodings to song titles

    '''

    # get list of all unique songs sorted alphabetically
    unique_songs = sorted(set(setlist_list))

    print(f'Phish has {len(unique_songs)} unique and {len(setlist_list)} total songs/set identifiers in this corpus.')
    print()
    
    # create a mapping for the encoded songs
    song_to_idx = {song:index for index, song in enumerate(unique_songs)}
    # add entry for <UNK> to handle new songs in the future
    song_to_idx['<UNK>'] = max(song_to_idx.values()) + 1
    # create reverse mapping
    idx_to_song = {v:k for k,v in song_to_idx.items()}
    
    return song_to_idx, idx_to_song

def encode_setlist_data(song_to_idx, setlist_list):
    '''
    Apply a song to index mapping to a list of all songs, but first randomly insert some unknown's for the model to learn from

    Args:
        song_to_idx (dict) - a mapping of song titles to numeric encodings
        setlist_list (list) - the object returned from the preprocess_data utility function
    
    Returns:
        encoded_setlist_list (list) - the input list with songs replaced by encodings

    '''

    # randomly insert the number of unique songs as unknowns - the model won't know each song one time...
    used_idx = []

    while len(used_idx) < len(song_to_idx):
        
        # pick a random song
        new_idx = random.randrange(len(setlist_list))
        
        if new_idx not in used_idx:
            
            if setlist_list[new_idx] not in ['<ENCORE>', '<ENCORE2>', '<SET1>', '<SET2>', '<SET3>', '<SET4>']:

                # overwrite it with <UNK>
                setlist_list[new_idx] = '<UNK>'
        
        # save used idx
        used_idx.append(new_idx)

    # encode the setlist
    encoded_setlist_list = [song_to_idx[song] for song in setlist_list]

    return encoded_setlist_list

def create_sequence_modeling_data(full_list, seq_length=100):
    '''
    Converts a list of items into a list consecutive sequences of length seq_length offset by one item, and
    then splits each sequence into X and y pairs where X is the first seq_length-1 items in the sequence, 
    and Y is the final item in the sequence 


    Args:
        full_list (list) - a list of data
        seq_length (int) - the desired length of X data

    Returns:
        X_data (ndarray) - 2D array representing X data of shape = number of sequences by length of sequences
        Y_data (ndarray) - 1D array representing y data for each sequence in X

    '''

    seq_length = seq_length
    sequences = []

    # create a list of sequences of length seq_length
    for i in range(seq_length, len(full_list)):
        # select the sequence of ints
        seq = full_list[i-seq_length: i+1]
        # append to list
        sequences.append(seq)

    # split sequences into X, y pairs where X is the first seq_length-1 
    # items in the sequence, and Y is the final song in the sequence

    sequences_array = np.array(sequences)
    X_data, y_data = sequences_array[:,:-1], sequences_array[:,-1]

    return X_data, y_data 
