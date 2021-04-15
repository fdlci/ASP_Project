import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.fft import ifft
from scipy.signal import get_window
from sklearn.metrics.pairwise import cosine_similarity
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa

def feat_matrix_librosa(audio, sr, hop_length=4096):
    """Computes the MFCC features using the Librosa library"""
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=20, fmax=sr/2, hop_length=hop_length, window='hamming')
    f = librosa.feature.mfcc(S=librosa.power_to_db(S),  n_mfcc=13)
    return f

def consecutive_sim(feat):
    """Get the cosine similarity of two consecutive frames"""
    sims = []
    for i in range(feat.shape[0]-1):
        v1 = feat[i].reshape(-1,1).T
        v2 = feat[i+1].reshape(-1,1).T
        sims.append(cosine_similarity(v1, v2)[0][0])
    return sims

def get_segmentation(frequence, sims, thresh):
    """Computes the ids of the feature vectors where a change of similarity
    occurs, i.e the similarity is lower than the thresh"""
    seg_time = []
    seg_ind = []
    for i, value in enumerate(sims):
        if value <= thresh:
            seg_time.append(i*frequence)
            seg_ind.append(i)
    return seg_time, seg_ind

def mean_feature_vectors(seg_ind, feat):
    """With the ids list from the segmentation, computes the mean of the 
    feature vectors associated with each other to create the potential states"""
    n, p = len(seg_ind), feat.shape[1]
    pot_states = np.zeros((n,p))
    previous_ind = None
    for i, ind in enumerate(seg_ind):
        if i == 0:
            pot_states[i] = np.mean(feat[:ind], axis = 0)
        else:
            #print(np.mean(feat[previous_ind:ind], axis=0).shape)
            pot_states[i] = np.mean(feat[previous_ind:ind], axis = 0)
        previous_ind = ind
    return pot_states

def grouping(pot_states, thresh):
    """Creates a dictionnary that indicates for all potential states which are the 
    ones that have a similarity higher than the threshold"""
    sim_pot_states = cosine_similarity(pot_states)
    n = sim_pot_states.shape[0]
    grouping = {}
    for i in range(n-1):
        grouping[i] = []
        for j in range(i+1, n):
            if sim_pot_states[i, j] >= thresh:
                grouping[i].append(j)
    grouping[n-1] = []
    return grouping

def merge_groups(groups):
    """According to the dictionnary created by grouping, we merge all the similar states
    together to get groups of potential states that by taking the mean will define the 
    initial states"""
    new_dict = {}
    visited_indices = []
    for i in groups:
        for j in groups[i]:
            visited_indices.append(j)
            if i not in visited_indices:
                if i in new_dict.keys():
                    new_dict[i] += groups[j]
                else:
                    new_dict[i] = groups[j]
        visited_indices.append(i)
    return new_dict

def initial_states(groups, pot_states):
    """Computes the initial states according to the groups formed by merge_groups"""
    initial_states = []
    for i in groups:
        mean_vec = np.zeros((pot_states.shape[1]))
        mean_vec += pot_states[i,:]
        for j in groups[i]:
            mean_vec += pot_states[j,:]
        initial_states.append(mean_vec/len(groups[i]))
    return np.array(initial_states)

def sudden_changes(labels):
    """This functions is used for the smoothing and computes the consecutive occurences of
    a given label"""
    current_label = labels[0]
    lab = []
    counter = 0
    c = 0
    while counter < len(labels):
        if labels[counter] == current_label:
            c += 1
            counter += 1
        else:
            lab.append((current_label, c))
            current_label = labels[counter]
            counter += 1
            c = 1
    lab.append((current_label, c))
    return lab

def smooth_function(changes, L):
    """Smooths the function obtained. If the sudden changes appear for less than 10 occurences,
    then the sudden changes are ignored."""
    new_labels = []
    previous_occ = changes[0][0]
    for value, occ in changes:
        if occ < L:
            new_labels += [previous_occ for i in range(occ)]
        else:
            new_labels += [value for i in range(occ)]
            previous_occ = value
    return new_labels

def computing_initial_states(feat, hop_length, sr, thresh=0.94):
    """Computes the initial states by appying the functions above"""

    # Compute the similarity of two consecutive feature vectors
    sims = consecutive_sim(feat.T)

    # Compute Segmentation
    seg_time, seg_ind = get_segmentation(hop_length/sr, sims, thresh)

    # Compute the potential states
    pot_states = mean_feature_vectors(seg_ind, feat.T)

    # Apply grouping
    groups = grouping(pot_states, thresh)
    new_groups = merge_groups(groups)

    # Compute the initial states
    states = initial_states(new_groups, pot_states)

    return pot_states, states


def plotting_figures(data, title, xlabel, ylabel):
    """Plots a figure with colorbar"""
    plt.figure(figsize=(12, 8))
    plt.imshow(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.set_cmap('hot_r')
    plt.colorbar(shrink=0.5)
    plt.show()

def plot_states_booth(data, data_smooth, time, title, xlabel, ylabel, sr=22050, hop_length=4096):
    """Plots a figure with the non smoothed and the smoothed version"""
    fig = plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, time, hop_length/sr), data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, time, hop_length/sr), data_smooth)
    plt.title("Smooth" + title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == '__main__':
    spring, sr = librosa.load('Spring.wav')
    spring_feat = feat_matrix_librosa(spring, sr)
    pot_states, states = computing_initial_states(spring_feat, hop_length=4096, sr=sr)
