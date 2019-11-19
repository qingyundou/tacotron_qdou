import numpy as np
import sys, os
sys.path.append('/home/dawna/tts/qd212/models/tacotron')

from hparams import hparams
hparams.parse('sample_rate=16000,frame_length_ms=20,frame_shift_ms=5,pml_dimension=163,spec_type=fwbnd')

from util import audio

# name = 'ljspeech-spec-00002'
# dirFile = '/scratch/je369/tacotron/163-lj-training/spectrograms/{}.npy'.format(name)
# dirFile = '/scratch/je369/tacotron/lj-training/{}.npy'.format(name)


# dirFile = '/home/dawna/tts/qd212/models/tacotron/results/logs-tacotron-bk2orig-asup/step-5-target-spec.npy'
# target_spectrogram = np.load(dirFile)
# log_dir = '/home/dawna/tts/qd212/models/tacotron/results/logs-tacotron-bk2orig-asup'
# target_waveform = audio.inv_spectrogram(target_spectrogram.T)
# audio.save_wav(target_waveform, os.path.join(log_dir, 'step-%d-target-audio.wav' % 23333333333))





# name = 'eval-0'
# dirFile_tmp = '/home/dawna/tts/qd212/models/tacotron/results/tacotron-bk2orig/eval/%s/%s'

# dirFile_tmp = '/home/dawna/tts/qd212/models/tacotron/results/tacotron-bk2orig/gta/%s/%s'
# dirFile_tmp = '/home/dawna/tts/qd212/models/tacotron/results/tacotron-bk2orig/eal/%s/%s'
# dirFile_tmp = '/home/dawna/tts/qd212/models/tacotron/results/tacotron-bk2orig/eval/%s/%s'

dirFile_tmp = '/home/dawna/tts/qd212/models/tacotron/results/tacotron-bk2orig-eal-scratch/eval/%s/%s'


os.makedirs(dirFile_tmp % ('wav',''), exist_ok=True)
# name_list = ['LJ001-0073','LJ001-0001','LJ001-0002','LJ001-0003']:
# name_list = ['eval-0','eval-1','eval-2','eval-3']:
# name_list = ['LJ001-0001','LJ001-0002','LJ001-0003']
name_list = ['LJ001-0073','LJ003-0229','LJ003-0296','LJ003-0304','LJ004-0208']
for name in name_list:
    dirFile_npy = dirFile_tmp % ('npy',name+'.npy')
    dirFile_wav = dirFile_tmp % ('wav',name+'.wav')
    target_spectrogram = np.load(dirFile_npy)
    target_waveform = audio.inv_spectrogram(target_spectrogram.T)
    audio.save_wav(target_waveform, dirFile_wav)
