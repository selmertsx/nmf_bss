import numpy as np
from pydub import AudioSegment

name = './data/row_onsei_josei.wav'
wav = AudioSegment.from_wav(name)
cutted_wav = wav[:30*1000]
cutted_wav.export("onsei_2.wav", format="wav")
