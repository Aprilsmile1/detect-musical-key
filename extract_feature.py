from matplotlib import pyplot as plt
import os
import seaborn as sns
import numpy as np
import pandas as pd
import hdf5_getters as GETTERS

def draw_pitch_chroma(file_path, segments):
    musical_keys = ["C", "Db", "D", "Eb" , "E" , "F", "F#", "G", "Ab", "A", "Bb", "B"]
    
    #test is 'ValidSongData/TRAAADZ128F9348C2E.h5'
    h5 = GETTERS.open_h5_file_read(file_path)
    song_title = str(GETTERS.get_title(h5))[1:]
    artist = str(GETTERS.get_artist_name(h5))[1:]
    original_pitches = GETTERS.get_segments_pitches(h5)
    h5.close()

    picture_pitches = np.transpose(original_pitches[:segments])
    picture_pitches = pd.DataFrame(data = picture_pitches, index = musical_keys)
    title = "First 100 chroma for " + song_title + " by " + artist

    plt.figure(figsize = (12, 4))
    sns.heatmap(picture_pitches, annot=False, xticklabels=10, cmap="viridis")
    plt.title(title)
    plt.show()
    return 0

#draw_pitch_chroma('ValidSongData/TRAABYW128F4244559.h5', 101)

#0 is minor and 1 is major
def relabel_feature(key, mode):
    if (key, mode) == (0, 1) or (key, mode) == (9, 0):
        feature_a = 0
    if (key, mode) == (1, 1) or (key, mode) == (10, 0):
        feature_a = 1
    if (key, mode) == (2, 1) or (key, mode) == (11, 0):
        feature_a = 2
    if (key, mode) == (3, 1) or (key, mode) == (0, 0):
        feature_a = 3
    if (key, mode) == (4, 1) or (key, mode) == (1, 0):
        feature_a = 4
    if (key, mode) == (5, 1) or (key, mode) == (2, 0):
        feature_a = 5
    if (key, mode) == (6, 1) or (key, mode) == (3, 0):
        feature_a = 6
    if (key, mode) == (7, 1) or (key, mode) == (4, 0):
        feature_a = 7
    if (key, mode) == (8, 1) or (key, mode) == (5, 0):
        feature_a = 8
    if (key, mode) == (9, 1) or (key, mode) == (6, 0):
        feature_a = 9
    if (key, mode) == (10, 1) or (key, mode) == (7, 0):
        feature_a = 10
    if (key, mode) == (11, 1) or (key, mode) == (8, 0):
        feature_a = 11
    return feature_a


def extract_feature_to_csv(data_path, csv_path):

    csv_column = ["file_name", "C",	"Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B", "ground_truth_key", "feature_a", "feature_b"]
    num_to_key = {0 : "C", 1 : "Db", 2 : "D", 3 : "Eb", 4 : "E", 5 : "F", 6 : "F#", 7 : "G", 8: "Ab", 9 : "A", 10 : "Bb", 11: "B"}
    feature = []

    file_list = os.listdir(data_path)
    for f in file_list:
        if f.endswith(".h5"):
            h5 = GETTERS.open_h5_file_read(os.path.join(data_path, f))
            original_pitches = GETTERS.get_segments_pitches(h5)
            key = GETTERS.get_key(h5)
            mode = GETTERS.get_mode(h5)
            h5.close()
            matrix_pitches = np.array(original_pitches)
            pitch_averages = np.mean(matrix_pitches, axis = 0)
            single_song_feature = list(pitch_averages)
            single_song_feature.insert(0, f)  
            single_song_feature.append(key)
            feature_a = relabel_feature(key, mode)
            single_song_feature.append(feature_a)
            single_song_feature.append(mode)

            feature.append(single_song_feature)
    csv_file = pd.DataFrame(data = feature, columns = csv_column)
    csv_file.to_csv('test_data_feature.csv')

extract_feature_to_csv("TestSongData", "")














