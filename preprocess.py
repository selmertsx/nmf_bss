import numpy as np
from pydub import AudioSegment

name = './data/onsei.wav'
wav = AudioSegment.from_wav(name)

cutted_wav = wav[:11*1000]
cutted_wav.export("onsei_cutted.wav", format="wav")
