import numpy as np
import matplotlib.pyplot as plt
import librosa
from mfcc_features import *


class MusicSummarization():
    """Computes a Music Summary of length L of an audio signal"""

    def __init__(self, sim_matrix, L, time):

        self.L = L
        self.sim_matrix = sim_matrix
        self.time = time
        self.time_values = np.linspace(0, self.time-L, num=len(self.sim_matrix)-self.L)
        # Compute the Qscore
        self.Qscore = self.scoreQ_list(self.sim_matrix, self.L)
        # Get the tuple with the Qscore and the length of the summary
        self.comb = (self.Qscore, self.L)
        
    def sum_column(self, matrix):
        """Sums over all columns of the similarity matrix to obtain a vector"""
        return np.sum(matrix, axis=1)

    def average_similarity(self, start, end, SumS):
        """Computes the average similarity according to the position of 
        the segment considered"""
        N = len(SumS)
        subS = SumS[start:end]
        return np.sum(subS) / (N*(end-start))

    def scoreQ(self, i, SumS):
        """Computes the average similarity of length L starting at i"""
        return self.average_similarity(i, i+self.L, SumS)

    def scoreQ_list(self, S, L):
        """Computes for all i the Qscore of the similarity matrix for 
        a given length L"""
        Q_list = []
        i_list = [i for i in range(len(S)-L)]
        SumS = self.sum_column(S)
        for i in i_list:
            Q_list.append(self.scoreQ(i, SumS))
        return Q_list

    def get_best_score(self):
        """Gets the argmax of the Qscore"""
        ind = np.argmax(self.Qscore)
        return self.time_values[ind]

    def plot_sum_sim(self, title):
        """Plots the column sum of the similarity matrix"""
        sim_sum = self.sum_column(self.sim_matrix)
        x = np.linspace(0, self.time, num=len(sim_sum))
        plt.plot(x, sim_sum)
        plt.title(title)
        plt.show()       

    def plotting_Qscore(self, Qscores):
        """Plots the Qscore for several lengths"""
        for Q, L in Qscores:
            x = np.linspace(0, self.time-L, num=len(self.sim_matrix)-L)
            plt.plot(x, Q, label=f'L = {L}')
        plt.legend()
        plt.title('Qscores for different values of the segment length L')
        plt.show()


if __name__ == '__main__':

    path_spring = 'Spring.wav'
    spring, sr = librosa.load(path_spring)
    title = 'Spring Sim Matrix'
    sim = SimilarityMatrix(spring, sr, title)

    List_scores_to_plot = []
    for L in [10, 20, 30]:
        music_sum = MusicSummarization(sim.similarity, L, sim.time)
        q = music_sum.get_best_score()
        print(f'Start{L}: {q}')
        List_scores_to_plot.append(music_sum.comb)

    music_sum.plotting_Qscore(List_scores_to_plot)




