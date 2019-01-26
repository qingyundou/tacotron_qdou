import numpy as np
from util import audio
from hparams import hparams

point_of_interest = '001'

# read from Nick
CMP = np.fromfile(f'/home/josh/tacotron/Nick/pml/herald_{point_of_interest}_1.cmp', dtype=np.float32)
# pml_features = nick_parameters.reshape()
f = open(f'/home/josh/tacotron/Nick/txt/herald_{point_of_interest}_1.txt', 'r')

# Load the audio to a numpy array:
wav = audio.load_wav(f'/home/josh/tacotron/Nick/wav/herald_{point_of_interest}_1.wav')

# Compute the linear-scale spectrogram from the wav:
spectrogram = audio.spectrogram(wav).astype(np.float32)

# try to reshape the Nick PML features into an 86 x something matrix
pml_dimension = 86
pml_features = CMP.reshape((-1, pml_dimension))

# Get the saved spectrogram
linear_target = np.load(f'/home/josh/tacotron/training/nick-spec-00605.npy')

print(f.read())
print('PML Feature Info', CMP.shape, CMP.size / pml_dimension, pml_features.shape)
print('Audio Info', spectrogram.shape, linear_target.shape, wav.shape)
