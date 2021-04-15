import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import librosa
from scipy.signal import get_window
from scipy.fft import fft

class MelScale():
    """Computes the Mel filterbank. We get the filters that will be used to compute the MFCC features.
    The implementation of the Mel scale was inspired by the following tutorial:
    https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial"""

    def __init__(self, fmin, fmax, num_filters):

        self.fmin = fmin
        self.fmax = fmax
        self.sr = 2*self.fmax
        self.num_filters = 40

        # get the delimitations of the intervals
        self.filter_points, self.mel_frequencies = self.get_filter_points(self.fmin, self.fmax)

        # get the Mel filters
        self.filters = self.get_filters(self.filter_points)

        # Normalize the filters
        self.normalization_factor = 2.0 / (self.mel_frequencies[2:self.num_filters+2] - self.mel_frequencies[:self.num_filters])
        self.filters = self.filters * self.normalization_factor[:, np.newaxis]

    def freq_to_mel(self, freq):
        """From the frequency domain to the mel domain"""
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def met_to_freq(self, mels):
        """From the mel domain to the frequency domain"""
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    def get_filter_points(self, fmin, fmax, mel_filter_num=40, FFT_size=2048):
        """Divides the frequency space into intervals of increasing size that will enable us to compute the 
        Mel filters"""
        
        fmin_mel = self.freq_to_mel(fmin)
        fmax_mel = self.freq_to_mel(fmax)
    
        mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
        freqs = self.met_to_freq(mels)
        
        return np.floor((FFT_size + 1) / self.sr * freqs).astype(int), freqs 

    def get_filters(self, filter_points, FFT_size=2048):
        """Computes the Mel filters according to the intervals computed previously. The Mel filters are a traingle
        of base width the size of the interval"""
        
        filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
        
        for n in range(len(filter_points)-2):
            filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
        
        return filters  

    def plotting_triangular_filter_bank(self):
        """Plots the Mel Filters"""

        filter_points = self.get_filter_points(self.fmin, self.fmax)[0]
        filters = self.get_filters(filter_points)
        
        plt.figure(figsize=(15,4))
        for filt in filters:
            plt.plot(filt)
        plt.title('Triangular Filter Bank')
        plt.show()

    def plotting_triangular_filter_bank_normalized(self, num_filters=40):
        """Plots the normalized Mel Filters"""
        
        filter_points, mel_frequencies = self.get_filter_points(self.fmin, self.fmax)
        filters = self.get_filters(filter_points)
        
        normalization_factor = 2.0 / (mel_frequencies[2:num_filters+2] - mel_frequencies[:num_filters])
        
        normalized_filters = filters * normalization_factor[:, np.newaxis]
        
        plt.figure(figsize=(15,4))
        for filt in normalized_filters:
            plt.plot(filt)
        plt.title('Normalized Triangular Filter Bank')
        plt.show()

class MFCC_features():
    """Computes the MFCC features according to the following steps:
            - Compute the frames of the audio signal
            - Window the frames with the Hamming window
            - Take the Fourier transform of all windows
            - Map the power spectrums onto the Mel Scale
            - Take the log of the filtered powers obtained
            - Apply the Discrete Cosine transform
        The implementation of the MFCC was inspired by the following tutorial:
        https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial"""

    def __init__(self, audio, dim_dct, fmin, fmax, num_filters):

        self.audio = audio

        # Computing the frames
        self.audio_frames = self.frames(audio)

        # Applying Hamming window
        self.hamming = get_window("hamming", Nx=2048, fftbins=True)
        self.hamming_frames = self.applying_hamming(self.audio_frames, self.hamming)

        # Apply fft
        self.fft_frames = self.apply_fft(self.hamming_frames)
        self.audio_spectrum = self.power_spectrum(self.fft_frames)

        # Mel Scale
        mel = MelScale(fmin, fmax, num_filters)
        self.filters = mel.filters

        # Filtering the signal
        self.audio_signal_processed = self.log_of_energy(self.filters, self.audio_spectrum)

        # DCT basis
        self.dct_filter_size = (dim_dct, num_filters)
        self.dct_basis = self.dct_filter(self.dct_filter_size)

        # MFCC features
        self.feat = self.mfcc_features(self.dct_basis, self.audio_signal_processed)

    def frames(self, x, frame_length=2048, hop_length=256):
        """Creates frames of the audio signal"""
        p = int((len(x)-frame_length)/hop_length)+1
        frames = np.zeros((frame_length, p))
        for i in range(0, len(x)-frame_length, hop_length):
            frames[:,int(i/hop_length)] = np.array([x[i:i+frame_length]]).T.reshape(-1)
        return frames.T    

    def applying_hamming(self, audio_frames, hamming):
        """Applies the Hamming window to each frame"""
        new_frames = np.zeros_like(audio_frames)
        for i, frame in enumerate(audio_frames):
            new_frames[i] = frame * hamming
        return new_frames

    def apply_fft(self, hamming_frames, FFT_size=2048):
        """Computes the FFT of all windowed frames"""
        hamming_frames = np.transpose(hamming_frames)

        audio_fft = np.empty((int(1 + FFT_size // 2), hamming_frames.shape[1]), dtype=np.complex64, order='F')
        for n in range(audio_fft.shape[1]):
            audio_fft[:, n] = fft(hamming_frames[:, n], axis=0)[:audio_fft.shape[0]]

        audio_fft = np.transpose(audio_fft)
        return audio_fft

    def power_spectrum(self, fft_signal):
        """Computes the power spectrum by taking the square of the module of the FFT"""
        return np.square(np.abs(fft_signal))

    def log_of_energy(self, normalized_filters, power):
        """Maps the powers onto the Mel Scale"""
        audio_filtered = np.dot(normalized_filters, np.transpose(power))
        return 10.0 * np.log10(audio_filtered)

    def dct_filter(self, dct_filter_size):
        """Computes the DCT basis"""     
        n, p = dct_filter_size
        dct_basis = np.zeros((n,p))
        dct_basis[0] = 1.0 / np.sqrt(p)
        samples = np.arange(1, 2*p, 2)
        
        for i in range(1, n):
            dct_basis[i] = np.sqrt(2/p) * np.cos(np.pi*i*samples/(2*p))
            
        return dct_basis

    def mfcc_features(self, dct_basis, audio_signal_processed):
        """Applies the DCT transform to the audio signal processed"""
        return np.dot(dct_basis, audio_signal_processed)

class SimilarityMatrix():
    """Computes the similarity matrix using the MFCC features"""

    def __init__(self, audio, sr, title, num_filters=40, dim_dct=13, fmax=11025, fmin=0):

        self.title = title

        # Taking out the zeros at the beginning of the signal to avoid mathematical errors
        self.audio = self.taking_out_initial_zeros(audio)

        # Compute the duration of the signal
        self.time = librosa.core.get_duration(self.audio, sr)

        # Compute the MFCC features
        self.feat = MFCC_features(self.audio, dim_dct, fmin, fmax, num_filters).feat

        # Compute the similarity matrix
        self.similarity = cosine_similarity(np.transpose(self.feat))

    def taking_out_initial_zeros(self, signal):
        """Takes out the zeros at the beginning of an audio signal because it can create frames with all zeros
        and can cause errors in the code"""
        for i in range(len(signal)):
            if signal[i] != 0:
                break
        return signal[i:]

    def plotting_the_similarity_matrix(self):
        """Plots the similarity matrix"""
        fig = plt.figure(figsize=(12, 8))
        plt.imshow(self.similarity, extent=[0,self.time,self.time,0])
        plt.title(self.title)
        plt.set_cmap('hot_r')
        plt.colorbar()
        plt.show()

if __name__ == '__main__':

    path_spring = 'Spring.wav'
    spring, sr = librosa.load(path_spring)
    title = 'Spring Sim Matrix'
    sim = SimilarityMatrix(spring, sr, title)
    sim.plotting_the_similarity_matrix()
