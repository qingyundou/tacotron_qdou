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

# System
import os
import scipy.io.wavfile

# Maths
import numpy as np

# CUED
#import sigproc
from .. import sigproc

def setbinpath(p):
    #print('STRAIGHT path: '+p)
    global STRAIGHTPATH
    STRAIGHTPATH = p
def getbinpath():
    global STRAIGHTPATH
    return STRAIGHTPATH
STRAIGHTPATH = ''

def isanalysiseavailable():
    import distutils.spawn
    mceppath = distutils.spawn.find_executable(os.path.join(STRAIGHTPATH, 'straight_mcep'))
    bndappath = distutils.spawn.find_executable(os.path.join(STRAIGHTPATH, 'straight_bndap'))
    return bool(mceppath) and bool(bndappath)

def issynthesisavailable():
    import distutils.spawn
    fftpath = distutils.spawn.find_executable(os.path.join(STRAIGHTPATH, 'synthesis_fft'))
    return bool(fftpath)
    
def isavailable():
    return isanalysiseavailable() and issynthesisavailable()

def sigmoid(x, a, b):
    s0 = 1.0 / (1.0 + np.exp(-1.0 * a * (0.0 - b)))
    s1 = 1.0 / (1.0 + np.exp(-1.0 * a * (1.0 - b)))

    return (1.0 / (1.0 + np.exp(-1.0 * a * (x - b))) - s0) / (s1 - s0)

def nbbndap(fs):
    '''
    Compute the number of aperiodicty bands.
    As needed by the vocoder STRAIGHT.
    
    Input
        fs: Sampling frequency
    '''

    nq = fs / 2

    fbark = 26.81 * nq / (1960 + nq ) - 0.53

    if fbark<2:
        fbark += 0.15*(2-fbark)
    if fbark>20.1:
        fbark +=  0.22*(fbark-20.1)

    numbands = int(np.round(fbark))

    return numbands

def analysis_spec(wav, fs, f0s, shift, dftlen, keeplen=False):
    '''
    Compute amplitude spectrogram using the STRAIGHT vocoder.

    fs     : [Hz]
    f0s    : [s,Hz]
    shift  : [s]
    dftlen : 
    '''

    #f0s = np.interp(np.arange(f0s[0,0], f0s[-1,0], shift), f0s[:,0], np.log(f0s[:,1])
    f0shift = np.median(np.diff(f0s[:,0]))

    tmprawwavfile = sigproc.gentmpfile('pystraight-analysis-spec.raw')
    tmpf0file = sigproc.gentmpfile('pystraight-analysis-spec.f0')
    tmpspecfile = sigproc.gentmpfile('pystraight-analysis-spec.spec')

    try:
        ((wav.copy()*np.iinfo(np.int16).max).astype(np.int16)).tofile(tmprawwavfile)

        np.savetxt(tmpf0file, f0s[:,1], fmt='%f')

        # Spectral envelope estimation
        cmd = os.path.join(STRAIGHTPATH, 'straight_mcep')+' -float -nmsg -f '+str(fs)+' -fftl '+str(dftlen)+' -apord '+str(dftlen/2+1)+' -shift '+str(shift*1000)+' -f0shift '+str(f0shift*1000)+' -f0file '+tmpf0file+' -pow -raw '+tmprawwavfile+' '+tmpspecfile # -nmsg
        ret = os.system(cmd)
        if ret>0: raise ValueError('ERROR during execution of straight_mcep')

        SPEC = np.fromfile(tmpspecfile, dtype='float32')
        SPEC = SPEC.reshape((-1, int(dftlen/2)+1))
    except:
        if os.path.exists(tmprawwavfile): os.remove(tmprawwavfile)
        if os.path.exists(tmpf0file): os.remove(tmpf0file)
        if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
        raise

    if os.path.exists(tmprawwavfile): os.remove(tmprawwavfile)
    if os.path.exists(tmpf0file): os.remove(tmpf0file)
    if os.path.exists(tmpspecfile): os.remove(tmpspecfile)

    if keeplen:
        if SPEC.shape[0]>f0s.shape[0]:
            SPEC = SPEC[:f0s.shape[0],:]
        elif SPEC.shape[0]<f0s.shape[0]:
            SPEC = np.vstack((SPEC, np.tile(SPEC[-1,:], (f0s.shape[0]-SPEC.shape[0], 1))))

    if 0:
        import matplotlib.pyplot as plt
        plt.ion()

        f, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        axs[0].imshow(sp.mag2db(SPEC.T), origin='lower', aspect='auto', interpolation='none', extent=(0, SPEC.shape[0]*shift, 0, fs/2))
        from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return SPEC

def analysis_aper(wav, fs, f0s, shift, dftlen):
    '''
    Compute features using the STRAIGHT vocoder.
    (STRAIGHT's Analysis stage)
    
    fs     : [Hz]
    f0s    : [s,Hz]
    shift  : [s]
    dftlen : 
    '''

    #f0s = np.interp(np.arange(f0s[0,0], f0s[-1,0], shift), f0s[:,0], np.log(f0s[:,1])
    f0shift = np.median(np.diff(f0s[:,0]))

    tmprawwavfile = sigproc.gentmpfile('pystraight-analysis-aper.raw')
    tmpf0file = sigproc.gentmpfile('pystraight-analysis-aper.f0')
    tmpaperfile = sigproc.gentmpfile('pystraight-analysis-aper.aper')

    try:
        ((wav.copy()*np.iinfo(np.int16).max).astype(np.int16)).tofile(tmprawwavfile)

        np.savetxt(tmpf0file, f0s[:,1], fmt='%f')

        # Aperiodicity estimation
        cmd = os.path.join(STRAIGHTPATH, 'straight_bndap')+' -float -nmsg -f '+str(fs)+' -shift '+str(shift*1000)+' -f0shift '+str(f0shift*1000)+' -f0file '+tmpf0file+' -fftl '+str(dftlen)+' -apord '+str(dftlen/2+1)+' -raw '+tmprawwavfile+' '+tmpaperfile # -nmsg
        ret = os.system(cmd)
        if ret>0: raise ValueError('ERROR during execution of straight_bndap')

        APER = np.fromfile(tmpaperfile, dtype='float32')
        APER = APER.reshape((-1, dftlen/2+1))
    except:
        if os.path.exists(tmprawwavfile): os.remove(tmprawwavfile)
        if os.path.exists(tmpf0file): os.remove(tmpf0file)
        if os.path.exists(tmpaperfile): os.remove(tmpaperfile)
        raise

    if os.path.exists(tmprawwavfile): os.remove(tmprawwavfile)
    if os.path.exists(tmpf0file): os.remove(tmpf0file)
    if os.path.exists(tmpaperfile): os.remove(tmpaperfile)

    if 0:
        import matplotlib.pyplot as plt
        plt.ion()

        plt.imshow(APER.T, origin='lower', aspect='auto', interpolation='none', extent=(0, APER.shape[0]*shift, 0, fs/2))
        from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return APER


def analysis(wav, fs, f0s, shift, dftlen):
    '''
    Compute features using the STRAIGHT vocoder.
    (STRAIGHT's Analysis stage)
    
    fs     : [Hz]
    f0s    : [s,Hz]
    shift  : [s]
    dftlen : 
    '''

    #f0s = np.interp(np.arange(f0s[0,0], f0s[-1,0], shift), f0s[:,0], np.log(f0s[:,1])
    f0shift = np.median(np.diff(f0s[:,0]))

    tmprawwavfile = sigproc.gentmpfile('pystraight-analysis.raw')
    tmpf0file = sigproc.gentmpfile('pystraight-analysis.f0')
    tmpspecfile = sigproc.gentmpfile('pystraight-analysis.spec')
    tmpaperfile = sigproc.gentmpfile('pystraight-analysis.aper')

    try:
        ((wav.copy()*np.iinfo(np.int16).max).astype(np.int16)).tofile(tmprawwavfile)

        np.savetxt(tmpf0file, f0s[:,1], fmt='%f')

        #print('STRAIGHTPATH='+STRAIGHTPATH)

        # Aperiodicity estimation
        cmd = os.path.join(STRAIGHTPATH, 'straight_bndap')+' -float -nmsg -f '+str(fs)+' -shift '+str(shift*1000)+' -f0shift '+str(f0shift*1000)+' -f0file '+tmpf0file+' -fftl '+str(dftlen)+' -apord '+str(dftlen/2+1)+' -raw '+tmprawwavfile+' '+tmpaperfile # -nmsg
        os.system(cmd)

        # Spectral envelope estimation
        cmd = os.path.join(STRAIGHTPATH, 'straight_mcep')+' -float -nmsg -f '+str(fs)+' -fftl '+str(dftlen)+' -apord '+str(dftlen/2+1)+' -shift '+str(shift*1000)+' -f0shift '+str(f0shift*1000)+' -f0file '+tmpf0file+' -pow -raw '+tmprawwavfile+' '+tmpspecfile # -nmsg
        os.system(cmd)

        SPEC = np.fromfile(tmpspecfile, dtype='float32')
        SPEC = SPEC.reshape((-1, dftlen/2+1))
        APER = np.fromfile(tmpaperfile, dtype='float32')
        APER = APER.reshape((-1, dftlen/2+1))
    except:
        if os.path.exists(tmprawwavfile): os.remove(tmprawwavfile)
        if os.path.exists(tmpf0file): os.remove(tmpf0file)
        if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
        if os.path.exists(tmpaperfile): os.remove(tmpaperfile)
        raise

    if os.path.exists(tmprawwavfile): os.remove(tmprawwavfile)
    if os.path.exists(tmpf0file): os.remove(tmpf0file)
    if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
    if os.path.exists(tmpaperfile): os.remove(tmpaperfile)

    if 0:
        import matplotlib.pyplot as plt
        plt.ion()

        f, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        axs[0].imshow(sp.mag2db(SPEC.T), origin='lower', aspect='auto', interpolation='none', extent=(0, SPEC.shape[0]*shift, 0, fs/2))
        axs[1].imshow(APER.T, origin='lower', aspect='auto', interpolation='none', extent=(0, SPEC.shape[0]*shift, 0, fs/2))
        from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return SPEC, APER


def synthesis(fs, f0s, SPEC, shift, APER=None, f0_forcecont=False, verbose=0):

    if f0s.shape[0]!=SPEC.shape[0] or ((not APER is None) and (SPEC.shape[0]!=APER.shape[0])):
        raise ValueError('Features dont have the same length (F0={}, SPEC={}, APER={}).'.format(f0s.shape, SPEC.shape, APER.shape))
    if (not APER is None) and (SPEC.shape[1]!=APER.shape[1]):
        raise ValueError('Spectral features dont have the same width (SPEC={}, APER={}).'.format(SPEC.shape, APER.shape))

    dftlen = (SPEC.shape[1]-1)*2

    if APER is None:
        APER = np.zeros(SPEC.shape)
        APER[f0s[:,1]>0,:int((float(dftlen)/fs)*4000.0)] = -100.0

    if f0_forcecont:
        # Replace zero values by interpolated values
        idx = f0s[:,1]>0
        f0s[:,1] = np.exp(np.interp(f0s[:,0], f0s[idx,0], np.log(f0s[idx,1])))

    tmpf0file = sigproc.gentmpfile('pystraight-synthesis.f0')
    tmpaperfile = sigproc.gentmpfile('pystraight-synthesis.aper')
    tmpspecfile = sigproc.gentmpfile('pystraight-synthesis.spec')
    tmpwavfile = sigproc.gentmpfile('pystraight-synthesis.wav')

    try:
        np.savetxt(tmpf0file, f0s[:,1], fmt='%f')
        APER.astype(np.float32).tofile(tmpaperfile)
        SPEC.astype(np.float32).tofile(tmpspecfile)

        cmd = os.path.join(STRAIGHTPATH, 'synthesis_fft')+' -f '+str(fs)+' -fftl '+str(dftlen)+' -spec -shift '+str(shift*1000)+' -apfile '+tmpaperfile+' -float '+tmpf0file+' '+tmpspecfile+' '+tmpwavfile+' ' #>/dev/null 2>&1
        if verbose==0:
            cmd += ' > /dev/null 2>&1'
        print(cmd)
        os.system(cmd)

        wavfs, wav = scipy.io.wavfile.read(tmpwavfile)
        wavdtype = wav.dtype
        wav = wav / float(np.iinfo(wavdtype).max)

    except:
        if os.path.exists(tmpf0file): os.remove(tmpf0file)
        if os.path.exists(tmpaperfile): os.remove(tmpaperfile)
        if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
        if os.path.exists(tmpwavfile): os.remove(tmpwavfile)
        raise

    if os.path.exists(tmpf0file): os.remove(tmpf0file)
    if os.path.exists(tmpaperfile): os.remove(tmpaperfile)
    if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
    if os.path.exists(tmpwavfile): os.remove(tmpwavfile)

    return wav
