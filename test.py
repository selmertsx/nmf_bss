import numpy as np
import librosa
import nimfa
import cmath

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from IPython.display import display, Audio

%matplotlib inline

sound_time_series, sampling_rate = librosa.load('data/doremi_sound.wav')
sound_frequency_domain = librosa.stft(sound_time_series, hop_length=256, win_length=512, n_fft=1024)
phase = np.angle(sound_frequency_domain)

K=4
nmf = nimfa.Nmf(np.abs(sound_frequency_domain), rank=K, max_iter=1000, update='divergence', seed="random_c", objective='div')

nmf_fit = nmf()
W = nmf_fit.basis()
H = nmf_fit.coef()
E = np.linalg.norm(nmf.residuals())

y = np.zeros([137728, 4])

for k in np.arange(K):
  XmagHat = np.dot(W[:,k], H[k,:])
  line_length, col_length = XmagHat.shape
  #位相成分を入れるための行列
  xhat = np.zeros([line_length, col_length], dtype=np.complex)

  # 振幅成分に位相成分を入れる
  for i in np.arange(line_length-1):
    for j in np.arange(col_length-1):
      xhat[i,j] = XmagHat[i,j] * cmath.exp(1j*phase[i,j])

  y_buf = librosa.istft(xhat, hop_length=256, win_length=512)
  display(Audio(y_buf, rate=sampling_rate))
  y[:, k] = y_buf

sum_y = np.sum(y, axis=1)
display(Audio( sum_y , rate=sampling_rate))
