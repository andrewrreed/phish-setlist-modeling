'''
Utility functions for the pyphish package

'''
import os
import pickle
from bs4 import BeautifulSoup



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

    print(f'Successfully pickled {obj} to {os.path.abspath(full_path)}')

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
    if file_path[-7:] != '.pickle':
        raise ValueError('The file must end with a .pickle suffix.')

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
