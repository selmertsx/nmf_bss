import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from IPython.display import display, Audio
import nimfa

np.set_printoptions(formatter={'float': '{:0.2f}'.format})

sound_time_series, sampling_rate = librosa.load('data/onsei_2.wav')
sound_frequency_domain = librosa.stft(sound_time_series)

K=60
basis_matrix, activate_matrix, cost_matrix = nimfa.Nmf(np.abs(sound_frequency_domain), R=K, n_iter=1000)

import pdb; pdb.set_trace()

