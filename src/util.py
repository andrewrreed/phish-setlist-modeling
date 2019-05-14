'''
Utility functions for the phish-setlist-modeling package

'''
import os
import pickle
import numpy as np
import pandas as pd
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
        setlist = row.setlistdata_clean.split(', ')
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
            setlist = row.setlistdata_clean + ', '

        # append to full list
        setlist_list.append(setlist)

    # join to one long string
    setlist_string = ''.join(setlist_list)
    
    
    ## ------- Create unique identifiers for sets/encores -------
    
    # replace ids
    setlist_string = setlist_string.replace('Set 1', '<SET1>').replace('Set 2', '<SET2>').replace('Set 3', '<SET3>').replace('Set 4', '<SET4>').replace('Encore 2', '<ENCORE2>').replace('Encore', '<ENCORE>')
    
    # split string data into list
    setlist_list = setlist_string.split(', ')

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
    
    # create a mapping and reverse mapping for the encoded songs
    song_to_idx = {song:index for index, song in enumerate(unique_songs)}
    idx_to_song = {v:k for k,v in song_to_idx.items()}
    
    return song_to_idx, idx_to_song

def encode_setlist_data(song_to_idx, setlist_list):
    '''
    Apply a song to index mapping to a list of all songs

    Args:
        song_to_idx (dict) - a mapping of song titles to numeric encodings
        setlist_list (list) - the object returned from the preprocess_data utility function
    
    Returns:
        encoded_setlist_list (list) - the input list with songs replaced by encodings

    '''

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


# ------------------------- General Utilities -------------------------

def create_pickle_object(obj, pickle_name, file_path='./pickle_objects/'):
    """Pickle a Python object and save to local directory

    Parameters
    ----------
    obj : Python object
        The Python object to pickle
    pickle_name : string
        The file name of that will be saved with the file; must end with '.pickle' suffix
    file_path : string
        The path (absolute or relative) to the target directory where the pickle file will be stored
    """

    # validate datatypes
    if isinstance(pickle_name, str) and isinstance(file_path, str):

        # verify .pickle suffix
        if pickle_name[-7:] != '.pickle':
            raise ValueError('The pickle_name argument must end with a .pickle suffix.')

        # build full path
        full_path = file_path + pickle_name
    else:
        raise ValueError('Both file_name and file_path arguments must be of type string.')

    # check if directory exists - create if it doesnt
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # open new file and dump serialized data
    pickle_out = open(full_path, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()

    print(f'Successfully pickled {pickle_name} to {os.path.abspath(full_path)}')

    return None

def load_pickle_object(file_path):
    """Load a pickled Python object from local directory

    Parameters
    ----------
    file_path : string
        The path (absolute or relative) to the target directory where the pickle file is stored

    Returns
    -------
    pickle_obj : Python object
        The Python object from stored serialized representation
    """

    # check if file_path is string
    if not isinstance(file_path, str):
        raise TypeError('The file_path argument must be a string.')

    # check if the file exists
    if not os.path.exists(file_path):
        raise NameError('The file or path provided does not exist.')

    # verify .pickle file as target
    if not (file_path[-7:] == '.pickle' or file_path[-4:] == '.pkl'):
        raise ValueError('The file must end with a .pickle or .pkl suffix.')

    pickle_in = open(file_path, 'rb')
    pickle_obj = pickle.load(pickle_in)

    return pickle_obj

def parse_setlist_field(html_response_string):
    """Parse the HTML response of a "setlist" field from the get_setlist() API wrapper

    Parameters
    ----------
    html_response_string : str
        The value stored in the "setlist" field from the get_setlist() API wrapper

    Returns
    -------
    full_setlist : str
        The show setlist as a string with songs/set identifiers separated by commas

    Example
    -------

    Input:
    '<p><span class=\'set-label\'>Set 1</span>: <a href=\'http://phish.net/song/you-enjoy-myself\' class=\'setlist-song\'>You Enjoy Myself</a>, <a href=\'http://phish.net/song/turtle-in-the-clouds\' class=\'setlist-song\'>Turtle in the Clouds</a>, <a href=\'http://phish.net/song/46-days\' class=\'setlist-song\'>46 Days</a>, <a href=\'http://phish.net/song/no-men-in-no-mans-land\' class=\'setlist-song\'>No Men In No Man\'s Land</a> > <a href=\'http://phish.net/song/emotional-rescue\' class=\'setlist-song\'>Emotional Rescue</a>, <a title="A quality jam with an uncommonly subdued tone seems to momentarily wind down at 5:15 before launching into another round of synth-groove and builds to a rocking finale capped by strong Trey leads. " href=\'http://phish.net/song/tube\' class=\'setlist-song\'>Tube</a> > <a href=\'http://phish.net/song/shade\' class=\'setlist-song\'>Shade</a>, <a href=\'http://phish.net/song/saw-it-again\' class=\'setlist-song\'>Saw It Again</a></p><p><span class=\'set-label\'>Set 2</span>: <a title="Massive Set 2 opener. Pivots into a gritty and nasty jam at Mike\'s urging, which gets very heavy (and a tad flubby) as Trey takes control. A gnarled groove emerges, with Page laying on the electric piano extra thick, and then Trey pivots into major-key territory. The jam grows warm and pleasant as Mike moves into the lead, Page throwing in a carnivalesque ambiance, then Trey starts firing off some crunchy solos and we enter the wonderful land of hard rocking. A few &quot;woos&quot; are goosed, then Page hits on some ascending not unlike &quot;Tweezer Reprise&quot; and the jam peaks in wonderful fashion. > into an equally outstanding &quot;Mercury&quot;." href=\'http://phish.net/song/set-your-soul-free\' class=\'setlist-song\'>Set Your Soul Free</a> > <a title="Great Version. > from a huge &quot;SYSF,&quot; play breaks free around 10:00, Trey\'s tone warm, rife with melody, floating atop waves of Page\'s now signature keys. With a series of licks Trey pushes play forward, the jam builds in energy, before a huge blast of sustain peels apart (with Page now on his piano) into one of &quot;Mercury\'s&quot; finer moments. Play rocks, with the band alighting upon a sudden, and infectious groove. Pretty cool. > &quot;Slave.&quot;    " href=\'http://phish.net/song/mercury\' class=\'setlist-song\'>Mercury</a> > <a href=\'http://phish.net/song/slave-to-the-traffic-light\' class=\'setlist-song\'>Slave to the Traffic Light</a> > <a href=\'http://phish.net/song/possum\' class=\'setlist-song\'>Possum</a> > <a href=\'http://phish.net/song/sanity\' class=\'setlist-song\'>Sanity</a> > <a href=\'http://phish.net/song/walk-away\' class=\'setlist-song\'>Walk Away</a></p><p><span class=\'set-label\'>Encore</span>: <a href=\'http://phish.net/song/more\' class=\'setlist-song\'>More</a>'

    Output:
    "Set 1, You Enjoy Myself, Turtle in the Clouds, 46 Days, No Men In No Man's Land, Emotional Rescue, Tube, Shade, Saw It Again, Set 2, Set Your Soul Free, Mercury, Slave to the Traffic Light, Possum, Sanity, Walk Away, Encore, More"

    """

    soup = BeautifulSoup(html_response_string, 'html.parser')

    # initialize list to collect songs
    full_setlist = []

    # loop through <p> tags to extract songs from each set (a <p> tag for each set)
    for set_ in list(soup.children):

        # verify the set is populated, if its unknown, skip it
        if str(set_) == '<i>Setlist unknown</i>':
            pass

        else:
            # verify only one set identifier per <p> tag
            if len(set_.find_all('span')) > 1:

                raise ValueError('Set data contains more than on span tag (i.e. set identifier). Need to investigate.')

            # extract set identifier (i.e. Set 1, Set 2, Set 3, Encore)
            set_identifier = set_.find_all('span')[0].get_text()

            # extract list of songs in order
            setlist = [song.get_text() for song in set_.find_all('a')]

            # combine
            combined_set = [set_identifier] + setlist

            full_setlist.extend(combined_set)

    return ', '.join(full_setlist)
