import os
import shutil
import sys
import time
import glob
import datetime
import sqlite3
import numpy as np
import hdf5_getters as GETTERS


#Prepare dataset
#Only data above a threshold of c = 0.5 for both key and mode were used, which limited the data to a subset of 3729 songs.
def select_valid_data(basedir,ext = '.h5'):
    threshold = 0.5
    count = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*'+ext))
        for f in files :
            h5 = GETTERS.open_h5_file_read(os.path.abspath(f))
            key_conf = GETTERS.get_key_confidence(h5)
            mode_conf = GETTERS.get_mode_confidence(h5)
            if key_conf >= threshold and mode_conf >= threshold:
                count += 1
                shutil.copy(os.path.abspath(f), 'ValidSongData')
                print(count)
            h5.close()
    return count 

# c = select_valid_data('MillionSongSubset')
# print(c)

            
