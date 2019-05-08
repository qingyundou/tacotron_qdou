'''
Copyright(C) 2016 Engineering Department, University of Cambridge, UK.

License
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Author
    Gilles Degottex <gad27@cam.ac.uk>
'''

import sys
import os
import numpy as np
import shutil
import glob
import datetime

import scipy.io

import sigproc as sp
from . import resampling

def avread(filename):
    '''
    Read any audio
    '''
    tmpoutfname = sp.gentmpfile('avread.wav')
    try:
        os.system('avconv -i '+filename+' '+tmpoutfname+' >/dev/null 2>&1')
        #os.system('avconv -i '+filename+' '+tmpoutfname)

        wav, fs, enc = wavread(tmpoutfname)
    except:
        if os.path.exists(tmpoutfname): os.remove(tmpoutfname)
        raise

    if os.path.exists(tmpoutfname): os.remove(tmpoutfname)

    return wav, fs, enc

def wavread(filename):
    if sp.use_pysndfile:
        wav, fs, enc = pysndfile.sndio.read(filename)
    else:
        fs, wav = scipy.io.wavfile.read(filename)
        if wav.dtype!='float32' and wav.dtype!='float16':
            wav = wav / float(np.iinfo(wav.dtype).max)
        enc = wav.dtype

    return wav, fs, enc

def wavgetfs(filename):
    if sp.use_pysndfile:
        _, fs, _ = pysndfile.sndio.read(filename)
    else:
        fs, _ = scipy.io.wavfile.read(filename)

    return fs

def wavwrite(filename, wav, fs, enc=None, norm_max_ifneeded=False, norm_max=False, verbose=0):

    if norm_max:
        wav_max = np.max(np.abs(wav))
        wav /= 1.05*wav_max

    elif norm_max_ifneeded:
        wav_max = np.max(np.abs(wav))
        if wav_max>=1.0:
            print('    WARNING: sigproc.wavwrite: waveform in file {} is clipping. Rescaling between [-1,1]'.format(filename))
            wav /= 1.05*wav_max

    if np.max(np.abs(wav))>=1.0:
        print('    WARNING: sigproc.wavwrite: waveform in file {} is clipping.'.format(filename))

    if sp.use_pysndfile:
        if enc==None: enc='pcm16'
        pysndfile.sndio.write(filename, wav, rate=fs, format='wav', enc=enc)
    else:
        if enc==None: enc = np.int16
        elif enc=='pcm16':
            enc = np.int16
        elif enc=='float32' or enc==dtype('float32'):
            raise ValueError('float not supported by scipy.io.wavfile')
        wav = wav.copy()
        wav = enc(np.iinfo(enc).max*wav)
        scipy.io.wavfile.write(filename, fs, wav)
    if verbose>0:
        print('    Output: '+filename)

def exportfile( srcf,               # Source file to export
                destf,              # Destination path to export to
                resample=None,      # [Hz] Resample the waveform the given frequency (e.g. 44100Hz).
                highpass_fcut=None, # [Hz] High-pass the waveform according to the given frequency
                normalize=None,     # [dB] Normalise the overall file amplitude to the given amplitude (e.g. -32dB)
                aligndelayref=None, # [filepath] Align temporally the source waveform to the given waveform file.
                usepcm16=False,     # Save the waveform using PCM16 sample format
                channelid=0         # Use only the first channel (left) if multiple channels are found.
                ):

    orifs = None

    if resample==None and normalize==None and usepcm16==False and aligndelayref==None and highpass_fcut==None:
        # Copy/Paste the original file, without normalization
        shutil.copy2(srcf, destf)
    else:
        wav, orifs, enc = wavread(srcf)
        if len(wav.shape)>1:
            wav = wav[:,channelid] # Keep only channelid in case multiple tracks are present.
        wavfs = orifs
        ##print("{:10.3f}".format(len(wav)/float(wavfs))+'s '+str(wavfs)+'Hz '+enc)
        if usepcm16:
            enc = 'pcm16'

        if resample!=None:
            wav = resampling.resample(wav, wavfs, resample)
            wavfs = resample

        if highpass_fcut!=None:
            (b, a) = scipy.signal.butter(4, highpass_fcut/(0.5*wavfs), btype='high')
            wav = scipy.signal.filtfilt(b, a, wav)

        if normalize!=None:
            wav_spn = sp.level_normalise(wav, wavfs, level=normalize, warn_onclip=False)
            # wav_sv56, _ = interfaces.sv56demo(wav, wavfs, level=normalize)

            if 0:
                import matplotlib.pyplot as plt
                plt.ion()
                plt.plot(wav, 'k')
                plt.plot(wav_sv56, 'b')
                plt.plot(wav_spn, 'r')
                from IPython.core.debugger import  Pdb; Pdb().set_trace()

            wav = wav_spn

        if aligndelayref!=None:
            # Re-load the first tag as reference
            refwav, refwavfs, refenc = wavread(aligndelayref)
            wav = sp.align_delay(wav, wavfs, refwav, refwavfs)

        wavwrite(destf, wav, fs=wavfs, enc=enc)

        return orifs

def waveforms_statistics(dirs):

    print('Scanning: {}'.format(dirs))

    fss = dict()
    durations = []

    for indir in dirs:
        print('Scanning {}'.format(indir))
        for infile in glob.glob(indir+'/*'):
            if os.path.isdir(infile): continue
            print('\r'+str(len(durations))+' files: '+os.path.basename(infile)+'                  \r'),
            sys.stdout.flush()
            try:
                # wavfs, wav = wavfile.read(infile)
                wav, wavfs, _ = wavread(infile)

                fss[wavfs] = 1

                durations.append(len(wav)/float(wavfs))
            except:
                pass

        print('\r                                                              \r'),

    durations = np.array(durations)

    print("Files: {}".format(len(durations)))
    print("Sampling frequency: {}".format(fss.keys()))
    print("Durations in [{:0.4},{:0.4}]s".format(np.min(durations), np.max(durations)))
    print("Total duration: {:0.4f}s ({})".format(np.sum(durations), datetime.timedelta(seconds=np.sum(durations))))
