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

import os
import subprocess
import numpy as np
import scipy.interpolate
from scipy import signal as sig

from lib import sigproc as sp
from . import fileio

def resample(wav, fs, trgfs, method=2, deterministic=True):
    '''
        deterministic [True] : Try to make it deterministic.
                               (e.g. sox (mehtod=2) is not deterministic by default)
                    ATTENTION This option has been tested only for method==2
    '''
    if method==1:
        # sndfile-resample (libresample)
        # 'c' argument
        #0 : Best Sinc Interpolator
        #1 : Medium Sinc Interpolator (default)
        #2 : Fastest Sinc Interpolator TO AVOID
        #3 : ZOH Interpolator TO AVOID
        #4 : Linear Interpolator TO AVOID
        # sndfile-resample _seems_ to be always deterministic

        tmpinfname = sp.gentmpfile('sndfile-resample-in.wav')
        tmpoutfname = sp.gentmpfile('sndfile-resample-out.wav')

        try:
            wavwrite(tmpinfname, wav, fs)

            cmd = 'sndfile-resample -c 0 -to '+str(trgfs)+' '+tmpinfname+' '+tmpoutfname
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
            #print(out)

            syn, synfs, synenc = wavread(tmpoutfname)
        except:
            if os.path.exists(tmpinfname): os.remove(tmpinfname)
            if os.path.exists(tmpoutfname): os.remove(tmpoutfname)
            raise

        if os.path.exists(tmpinfname): os.remove(tmpinfname)
        if os.path.exists(tmpoutfname): os.remove(tmpoutfname)

    elif method==2:
        # SOX
        # VHQ: -v -s: The fastest with the results among the bests
        # ATTENTION:If deterministic=False, sox is NOT deterministic!
        #           I.e. it does NOT produce the same samples for each run!

        tmpinfname = sp.gentmpfile('sox-resample-in.wav')
        tmpoutfname = sp.gentmpfile('sox-resample-out.wav')

        try:
            fileio.wavwrite(tmpinfname, wav, fs)

            cmd = 'sox '
            if deterministic: cmd += '--no-dither '
            cmd += tmpinfname+' '+tmpoutfname+' rate -v -s '+str(trgfs)
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
            #print(out)

            syn, synfs, synenc = fileio.wavread(tmpoutfname)
        except:
            if os.path.exists(tmpinfname): os.remove(tmpinfname)
            if os.path.exists(tmpoutfname): os.remove(tmpoutfname)
            raise

        if os.path.exists(tmpinfname): os.remove(tmpinfname)
        if os.path.exists(tmpoutfname): os.remove(tmpoutfname)

    elif method==3:
        '''
        Resample using FFT and power of 2
        Create sometimes a significant peak at Nyquist
        '''
        syn = wav.copy()
        wavlen = syn.shape[0]
        wavlenpow2  = int(np.power(2, np.floor(np.log2(wavlen))+1))
        syn = np.pad(syn, (0, wavlenpow2-wavlen), constant_values=(0,0), mode='constant')
        syn = scipy.signal.resample(syn, np.round(len(syn)*float(trgfs)/fs))
        syn = syn[:np.round(wavlen*float(trgfs)/fs)]

    if 0:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.plot(np.arange(len(wav))/float(fs), wav, 'k')
        plt.plot(np.arange(len(syn))/float(trgfs), syn, 'b')
        from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return syn

# Resample feature using the nearest
def featureresample(ts, X, nts):
    if len(X.shape)>1:
        Y = np.zeros((len(nts), X.shape[1]))
    else:
        Y = np.zeros(len(nts))
    for n, t in enumerate(nts):
        idx = np.where(ts>=t)[0]
        if len(idx)==0:
            idx = X.shape[0]-1
        else:
            idx = np.min(idx) # Nearest
        idx = np.clip(idx, 0, X.shape[0]-1)
        if len(X.shape)>1:
            Y[n,:] = X[idx,:]
        else:
            Y[n] = X[idx]
    return Y

def f0s_resample_pitchsync(f0s, nbperperiod, f0min=20.0, f0max=5000.0):
    f0s = f0s.copy()

    # Interpolate where there is zero values
    f0s[:,1] = np.interp(f0s[:,0], f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])

    f0s[:,1] = np.clip(f0s[:,1], f0min, f0max)

    ts = [0.0]
    while ts[-1]<f0s[-1,0]:
        cf0 = np.interp(ts[-1], f0s[:,0], f0s[:,1])
        ts.append(ts[-1]+(1.0/nbperperiod)/cf0)
    f0s = np.vstack((ts, np.interp(ts, f0s[:,0], f0s[:,1]))).T

    return f0s

def f0s_resample_cst(f0s, timeshift):
    f0s = f0s.copy()

    vcs = f0s.copy()
    vcs[vcs[:,1]>0,1] = 1

    nts = np.arange(f0s[0,0], f0s[-1,0], timeshift)

    # The voicing resampling has to be done using nearest ...
    vcsfn = scipy.interpolate.interp1d(vcs[:,0], vcs[:,1], kind='nearest', bounds_error=False, fill_value=0)

    # ... whereas the frequency resampling need linear interpolation, while ignoring the voicing
    f0s = np.interp(nts, f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])

    # Put back the voicing
    f0s[vcsfn(nts)==0] = 0.0

    f0s = np.vstack((nts, f0s)).T

    if 0:
        plt.plot(f0s[:,0], f0s[:,1])

    return f0s

def f0s_rmsteps(f0s):
    '''
    Description
        Removes steps in the F0 curve.

        Steps can come from some F0 estimator (e.g. those based on GCI
        detection are likely to exhibits these).

        For pulse synthesis, it avoids some glitches around the main lobes

        It might be bad for creaky voice (oversmoothing the f0 curve),
        though F0 estimate in creaky voice is quite likely to be wrong anyway.
    '''
    f0sori = f0s.copy()
    f0s = f0s.copy()
    voicedi = np.where(f0s[:,1]>0)[0]
    shift = np.mean(np.diff(f0s[:,0]))
    fc = (1.0/shift)/4.0  # The cut-off frequency
    hshift = (1.0/fc)/8.0 # The high sampling rate for resampling the original curve
    data = np.interp(np.arange(0.0, f0s[-1,0], hshift), f0s[voicedi,0], f0s[voicedi,1])
    b, a = sig.butter(8, fc/(0.5/hshift), btype='low')
    f0ss = sig.filtfilt(b, a, data)
    f0s[voicedi,1] = np.interp(f0s[voicedi,0], np.arange(0.0, f0s[-1,0], hshift), f0ss)

    if 0:
        plt.plot(f0sori[:,0], f0sori[:,1], 'k')
        plt.plot(f0s[:,0], f0s[:,1], 'b')
        from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return f0s
