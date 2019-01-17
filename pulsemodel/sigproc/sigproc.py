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
import errno
import getpass
import uuid
import numpy as np
import warnings

import misc
import resampling

use_pysndfile = False
if use_pysndfile:
    import pysndfile
else:
    import scipy.io.wavfile

def settmppath(p):
    global TMPPATH
    TMPPATH = p
    print('sigproc: Temporary directory: '+TMPPATH)

if os.path.exists('/scratch') and os.path.isdir('/scratch'):
    TMPPATH='/scratch'
elif os.path.exists('/dev/shm') and os.path.isdir('/dev/shm'):
    TMPPATH='/dev/shm'
else:
    TMPPATH='/tmp'

def gentmpfile(name):
    tmpdir = os.path.join(TMPPATH,getpass.getuser())
    misc.makedirs(tmpdir)
    tmpfile = os.path.join(tmpdir,'sigproc.pid%s.%s.%s' % (os.getpid(), str(uuid.uuid4()), name))
    return tmpfile

def setbinpath(p):
    global BINPATH
    BINPATH = p
    #print('sigproc binary directory: '+BINPATH)
def getbinpath():
    global BINPATH
    return BINPATH
BINPATH = ''

# Conversions ------------------------------------------------------------------

def mag2db(a):
    return 20.0*np.log10(np.abs(a))

def lin2db(a):
    return log2db(np.log(np.abs(a)))

def db2lin(a):
    return np.exp(db2log(a))

def log2db(a):
    return 20.0*a/np.log(10.0)

def db2mag(d):
    return 10.0**(d/20.0)

def wrap(p):
    return np.angle(np.exp(1j*p))

def lin2mel(f):
    return 1125.0 * np.log(1.0 + f/700.0)

def mel2lin(m):
    return 700.0 * (np.exp(m/1125.0) - 1.0)


# Misc sigproc functions -------------------------------------------------------

def spec_ener(S):
    dftlen = (len(S)-1)*2
    return np.sqrt((S[0]**2+2*np.sum(abs(S[1:-1])**2)+S[-1]**2)/dftlen)

# delay [samples]
def spec_delay(delay, dftlen):
    return np.exp((delay*2j*np.pi/dftlen)*np.arange(dftlen/2+1))


# Circular mean from phase gravity center
def gphi2circmean(v):
    return np.angle(v)

# Circular variance from phase gravity center
def gphi2circstd(v):
    return np.sqrt(-2*np.log(abs(v)))

# Circular variance
def spec_circmean(S):
    S = S.copy()
    S[abs(S)==0] = np.finfo(S.dtype).tiny
    S /= abs(S)
    v = np.mean(np.real(S)) + 1j*np.mean(np.imag(S))
    return gphi2circmean(v)

# Circular variance
def spec_circstd(S):
    S = S.copy()
    S[abs(S)==0] = np.finfo(S.dtype).tiny
    S /= abs(S)
    v = np.mean(np.real(S)) + 1j*np.mean(np.imag(S))
    return gphi2circstd(v)

def butter2hspec(fc, o, fs, dftlen, high=False):
    '''
    Supposed to be the amplitude response of a Butterworth filter
    fc: cut-off [Hz]
    o: order
    fs: sampling frequency [Hz]
    '''

    F = fs*np.arange(dftlen/2+1)/dftlen
    H = 1.0/np.sqrt(1.0 + (F/fc)**(2*o))

    if high:
        H = 1.0-H

    return H

def hspec2minphasehspec(X, replacezero=False):
    if replacezero:
        X[X==0.0] = np.finfo(X[0]).resolution
    dftlen = (len(X)-1)*2
    cc = np.fft.irfft(np.log(X))
    cc = cc[:dftlen/2+1]
    cc[1:-1] *= 2
    return np.exp(np.fft.rfft(cc, dftlen))

def hspec2spec(X):
    return np.hstack((X, X[-2:0:-1]))

def framesignal(wav, fs, t, winlen):
    # Extract the signal segment to analyse
    nt = int(round(fs*t))
    winidx = nt + np.arange(-int((winlen-1)/2),int((winlen-1)/2)+1)
    if winidx[0]<0 or winidx[-1]>=len(wav):
        # The window is partly outside of the signal ...
        wav4win = np.zeros(winlen)
        # ... copy only the existing part
        itouse = np.logical_and(winidx>=0,winidx<len(wav))
        wav4win[itouse] = wav[winidx[itouse]]
    else :
        wav4win = wav[winidx]

    return wav4win, nt

def active_signal_level(wav, fs, speechthreshbelowmax=24):
    '''
    It does _not_ follow ITU-T Rec. G.191 of the "Active Speech Level" !
    It should, however, provide a value with similar properties:
        * Robust to noise presence (up to noise lvl below -24dB below max time amplitude envelope)
        * Robust to clicks
    '''

    # Get envelope that is not time shifted wrt signal
    (b, a) = scipy.signal.butter(4, 10.0/(0.5*fs), btype='low')
    env = scipy.signal.filtfilt(b, a, abs(wav))
    envdb = mag2db(env)

    # Max env value. HYP: it is robust enough egainst clicks
    envmax = np.max(envdb)
    actlvl = envmax

    silence_thresh = envmax-speechthreshbelowmax    # Measure active level in a X dB range below maximum env amplitude, HYP: The noise floor is below silence_thresh
    actlvl = np.mean(envdb[envdb>silence_thresh])

    # The max of the env is already robust against additive noise, so skip the rest
    # Estimate the noise floor
    # b = np.hanning(int(fs*0.020))
    # b /= np.sum(b)
    # noiseenv = scipy.signal.filtfilt(b, 1.0, abs(wav))
    # noisefloor = np.min(mag2db(noiseenv[len(b):-len(b)]))+6 # HYP: Noise floor is 6dB above minimum
    # silence_thresh = envmax-32.0    # Measure active level in a 32 dB range below maximum env amplitude
    # actlvl = np.mean(envdb[envdb>silence_thresh])

    # Histogram
    # [H, He] = np.histogram(envdb, bins=1000, range=[-150, 0.0], density=None)

    if 0:
        import matplotlib.pyplot as plt
        plt.ion()
        # plt.subplot(211)
        plt.plot(wav, 'k')
        plt.plot(env, 'b')
        # plt.plot(noiseenv, 'r')
        plt.plot(envdb, 'b')
        # plt.plot(mag2db(noiseenv), 'r')
        plt.plot([0, len(wav)], envmax*np.ones(2), 'k')
        plt.plot([0, len(wav)], silence_thresh*np.ones(2), 'b')
        plt.plot([0, len(wav)], actlvl*np.ones(2), 'g')
        # plt.plot([0, len(wav)], noisefloor*np.ones(2), 'r')
        # plt.subplot(212)
        # plt.plot((He[:-1]+He[1:])*0.5, H, 'b')
        from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return actlvl

def level_normalise(wav, fs, level=-26, warn_onclip=True):

    actlvl = active_signal_level(wav, fs)

    wavnormed = wav*db2mag(level-actlvl)

    if warn_onclip and np.max(abs(wavnormed))>=1.0:
        print('    WARNING: sigproc.level_normalise: The waveform is clipping after normalisation!')

    return wavnormed

def align_delay(wav, fs, refwav, reffs):
    if reffs!=fs:
        refwav = resampling.resample(refwav, reffs, fs)
        reffs = fs

    # Compute energy envelopes
    (b, a) = scipy.signal.butter(4, 50.0/(0.5*fs), btype='low')
    nrg = np.exp(scipy.signal.filtfilt(b, a, np.log(np.abs(wav)+1e-12)))
    refnrg = np.exp(scipy.signal.filtfilt(b, a, np.log(np.abs(refwav)+1e-12)))

    # Normalize
    nrg -= np.mean(nrg)
    nrg /= np.std(nrg)
    refnrg -= np.mean(refnrg)
    refnrg /= np.std(refnrg)

    # Compute cross-correlation
    dftlen = 2**(1+int(np.log2(np.max((len(nrg), len(refnrg))))))
    NRG = np.fft.rfft(nrg, dftlen)
    REFNRG = np.fft.rfft(refnrg, dftlen)
    CC = np.conj(NRG)*REFNRG
    cc = np.fft.fftshift(np.fft.irfft(CC))

    # Get the delay
    delayi = np.argmax(cc)-dftlen/2

    if delayi<0:
        aligned = wav[int(-delayi):]
    elif delayi>0:
        aligned = np.insert(wav.copy(), 0, wav[0]*np.ones(int(delayi)))
    else:
        aligned = wav.copy()

    # Fix the size to the reference
    if len(aligned)<len(refwav):
        aligned = np.append(aligned, aligned[-1]*np.ones(len(refwav)-len(aligned)))

    aligned = aligned[:len(refwav)] # Cut anything after the reference size

    if 0:
        plt.plot(refwav, 'k')
        plt.plot(wav, 'b')
        plt.plot(aligned, 'r')
        from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return aligned
