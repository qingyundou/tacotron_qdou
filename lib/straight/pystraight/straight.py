# System
import sys
import os
#import argparse
#import subprocess
#import re
#import pickle
import scipy.io.wavfile

# Maths
import numpy as np
#import scipy.signal

## CUED
import sigproc

import matplotlib.pyplot as plt
plt.ion()

def setbinpath(p):
    print('STRAIGHT path: '+p)
    global STRAIGHTPATH
    STRAIGHTPATH = p
STRAIGHTPATH = ''

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

def bndap2aper(BAP, fs, dftlen=4096, sigp=1.2):
    '''
        Calc number of critical bands and their edge frequency from sampling frequency
        Added by J. Yamagishi (28 March 2010)
        Reference 
        http://en.wikipedia.org/wiki/Bark_scale
        
        Zwicker, E. (1961), "Subdivision of the audible frequency range
        into critical bands," The Journal of the Acoustical Society of
        America, 33, Feb., 1961.
        
        H. Traunmuller (1990) "Analytical expressions for the tonotopic
        sensory scale" J. Acoust. Soc. Am. 88: 97-100.      
        
        Re-written in python by G. Degottex gad27@cam.ac.uk
    '''

    numbands = nbbndap(fs)

    if BAP.shape[1]!=nbbndap(fs):
        print('WARNING: feature size and expected number of aperiodicity bands dont match!')

    #print("BAP shape: "+str(BAP.shape))

    WNZ = np.zeros((BAP.shape[0], dftlen/2+1))
    nq = fs/2.0

    for lii in np.arange(BAP.shape[0]):
        for b in np.arange(BAP.shape[1]):
            #print('flop')
            if numbands != BAP.shape[1]:
                # fixed conventional frequency bands              
                if b == 0:
                    bs = 0
                    be = dftlen / 16
                elif b == 1:
                    bs = dftlen / 16
                    be = dftlen / 8
                elif b == 2:
                    bs = dftlen / 8
                    be = dftlen / 4
                elif b == 3:
                    bs = dftlen / 4
                    be = dftlen * 3 / 8
                else:
                    bs = dftlen * 3 / 8
                    be = dftlen / 2 + 1

            else:
                # critical-band-limited aperiodicity 
                startf = 1960 / (26.81 / (b + 0.53) - 1)
                startf = round (startf / 100 ) * 100    
                endf = 1960 / (26.81 / (b + 1 + 0.53) - 1)
                endf = round (endf / 100 ) * 100
            
                # deltaf = sampling frequency / dftlen
                bs = startf * dftlen / fs
                if endf < nq:
                    be = endf * dftlen / fs
                else:
                    be = dftlen / 2 + 1

            # weighting for mixed excitation
            if sigp <= 0.0:
                wnzv = np.min((1.0, np.power(10.0, BAP[lii,b]/20.0)))
            else:
                wnzv = sigmoid(np.power(10.0, BAP[lii,b]/20.0), sigp, 0.25)
                wnzv = np.min((1.0, wnzv))

            for k in np.arange(bs,be):
                WNZ[lii,k] = wnzv
                #if k!=0 and k!=dftlen/2:
                #wnz[lii, dftlen-k] = wnzv

    WNZ = 20*np.log10(WNZ)

    return WNZ


def aper2bndap(APER, fs):

    '''
        Calc number of critical bands and their edge frequency from sampling frequency
        Added by J. Yamagishi (28 March 2010)
        Reference 
        http://en.wikipedia.org/wiki/Bark_scale
        
        Zwicker, E. (1961), "Subdivision of the audible frequency range
        into critical bands," The Journal of the Acoustical Society of
        America, 33, Feb., 1961.
        
        H. Traunmuller (1990) "Analytical expressions for the tonotopic
        sensory scale" J. Acoust. Soc. Am. 88: 97-100.      
        
        Re-written in python by G. Degottex gad27@cam.ac.uk
    '''
    
    # Calc the number of critical bands required for sampling frequency
    numbands = nbbndap(fs)

    dftlen = (APER.shape[1]-1)*2

    BNDAP = np.zeros((APER.shape[0], numbands))

    for b in np.arange(numbands):
        # critical-band-limited aperiodicity 
        startf = 1960 / (26.81 / (b + 0.53) - 1)
        startf = round (startf / 100 ) * 100    
        endf = 1960 / (26.81 / (b + 1 + 0.53) - 1)
        endf = round (endf / 100 ) * 100

        if (startf < 20.0):
            startf = 20  # human hearing (20 Hz - 20kHz)

        # deltaf = sampling frequency / args.outfftl 
        bs = startf * dftlen / fs
        if endf < fs/2.0:
            be = endf * dftlen / fs
        else:
            be = dftlen / 2 + 1

        for n in np.arange(APER.shape[0]):
            BNDAP[n,b] = np.mean(APER[n,int(bs):int(be)])

    return BNDAP

def aperiodicity(wav, fs, f0s, dftlen, ams=None, outresidual=False):
    '''
        Computing the overall harm signal first, then estimating the noise from the residual.
        It should be a more accurate way to compute the aperiodicity than the
        original STRAIGHT's implementation
    '''
    from lib.sigproc import sinusoidal

    # Computing the overall harm signal first, then estimating the noise from the residual
    #sins = sinusoidal.estimate_sinusoidal_params(wav, fs, f0s)
    #wavlen = len(wav)
    #sinwav = sinusoidal.synthesize_harmonics(f0s, sins, fs, wavlen)
    #res = wav-sinwav

    # Replace 0s by interpolations
    f0s = f0s.copy()
    f0s[:,1] = np.interp(f0s[:,0], f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])
    
    if outresidual:
        ress = np.zeros(wav.shape)
        reswins = np.zeros(wav.shape)

    F = (float(fs)/dftlen)*np.arange(dftlen/2+1)
    APER = np.zeros((len(f0s[:,0]),dftlen/2+1))
    for n, t in enumerate(f0s[:,0]):
        f0 = f0s[n,1]
        print "\rt={:0.3f}s({:.0f}%) f0={:0.2f}Hz".format(t, 100*t/f0s[f0s.shape[0]-1,0], f0),

        # Window's length
        winlen = int(0.5+(3*fs/f0)/2)*2+1 # with rounding

        # Extract the signal segment to analyse
        winidx = np.arange(-int((winlen-1)/2),int((winlen-1)/2+1), dtype=np.int64)
        winidx += int(0.5+fs*t)
        if winidx[0]<0 or winidx[-1]>=len(wav):
            # The window is partly outside of the signal ...
            wav4win = np.zeros(winlen)
            # ... copy only the existing part
            itouse = np.logical_and(winidx>=0,winidx<len(wav))
            wav4win[itouse] = wav[winidx[itouse]]
        else :
            wav4win = wav[winidx]

        # The initial frequencies are
        freqs = f0 * np.arange(int(np.floor((fs/2.0-f0/2.0)/f0))+1)

        if np.linalg.norm(wav4win)<sys.float_info.epsilon:
            # The signal is empty: Add "empty" data
            # TODO
            continue

        # Window's shape
        win = np.blackman(winlen)                 
        win = win/sum(win) # Normalize for sinusoidal content

        S = sinusoidal.compute_dft(wav4win, fs, win, dftlen, winidx, ams)

        sin = sinusoidal.extract_peaks(S, fs, f0, winlen, dftlen)

        #from IPython.core.debugger import  Pdb; Pdb().set_trace()
        syn = sinusoidal.synthesize_harmonics(np.array([[((winlen-1)/2.0)/float(fs), f0]]), [sin], fs, winlen)

        res = wav4win-syn

        if winidx[0]<0 or winidx[-1]>=len(wav):
            # The window is partly outside of the signal ...
            # ... copy only the existing part
            itouse = np.logical_and(winidx>=0,winidx<len(wav))
            ress[winidx[itouse]] += res[itouse]*win[itouse]
            reswins[winidx[itouse]] += win[itouse]
        else:
            ress[winidx] += res*win
            reswins[winidx] += win

        N = sp.mag2db(np.fft.rfft(res*win, dftlen))

        E = np.interp(F, sin[0,1:], sp.mag2db(sin[1,1:]))

        APER[n,:] = N - E

        if t>0.3 and 0:
            SA = sp.mag2db(S)
            SA[np.isinf(SA)] = np.finfo(SA[0]).min
            plt.plot(F, SA, 'k')
            plt.plot(sin[0,:], sp.mag2db(sin[1,:]), 'xk')
            SYN = sinusoidal.compute_dft(syn, fs, win, dftlen, winidx)
            plt.plot(F, sp.mag2db(SYN), 'b')
            plt.plot(F, E, 'b')
            plt.plot(F, sp.mag2db(np.fft.rfft(res*win, dftlen)), 'r')
            plt.plot(F, APER[n,:], 'g')
            from IPython.core.debugger import  Pdb; Pdb().set_trace()

    print '\r                                                               \r',

    if outresidual:
        idx = reswins>0.0
        ress[idx] /= reswins[idx]
        return APER, ress
    else:
        return APER

def analysis_spec(wav, fs, f0s, shift, dftlen):
    '''
    Compute amplitude spectrogram using the STRAIGHT vocoder.

    fs     : [Hz]
    f0s    : [s,Hz]
    shift  : [s]
    dftlen : 
    '''

    #f0s = np.interp(np.arange(f0s[0,0], f0s[-1,0], shift), f0s[:,0], np.log(f0s[:,1])
    f0shift = np.median(np.diff(f0s[:,0]))

    tmprawwavfile = sigproc.gentmpfile('straight-analysis-spec.raw')
    tmpf0file = sigproc.gentmpfile('straight-analysis-spec.f0')
    tmpspecfile = sigproc.gentmpfile('straight-analysis-spec.spec')

    try:
        ((wav.copy()*np.iinfo(np.int16).max).astype(np.int16)).tofile(tmprawwavfile)

        np.savetxt(tmpf0file, f0s[:,1], fmt='%f')

        # Spectral envelope estimation
        cmd = STRAIGHTPATH+'straight_mcep -float -nmsg -f '+str(fs)+' -fftl '+str(dftlen)+' -apord '+str(dftlen/2+1)+' -shift '+str(shift*1000)+' -f0shift '+str(f0shift*1000)+' -f0file '+tmpf0file+' -pow -raw '+tmprawwavfile+' '+tmpspecfile # -nmsg
        ret = os.system(cmd)
        if ret>0: raise ValueError('ERROR during execution of straight_mcep')

        SPEC = np.fromfile(tmpspecfile, dtype='float32')
        SPEC = SPEC.reshape((-1, dftlen/2+1))
    except:
        if os.path.exists(tmprawwavfile): os.remove(tmprawwavfile)
        if os.path.exists(tmpf0file): os.remove(tmpf0file)
        if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
        raise

    if os.path.exists(tmprawwavfile): os.remove(tmprawwavfile)
    if os.path.exists(tmpf0file): os.remove(tmpf0file)
    if os.path.exists(tmpspecfile): os.remove(tmpspecfile)

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

    tmprawwavfile = sigproc.gentmpfile('straight-analysis-aper.raw')
    tmpf0file = sigproc.gentmpfile('straight-analysis-aper.f0')
    tmpaperfile = sigproc.gentmpfile('straight-analysis-aper.aper')

    try:
        ((wav.copy()*np.iinfo(np.int16).max).astype(np.int16)).tofile(tmprawwavfile)

        np.savetxt(tmpf0file, f0s[:,1], fmt='%f')

        # Aperiodicity estimation
        cmd = STRAIGHTPATH+'straight_bndap -float -nmsg -f '+str(fs)+' -shift '+str(shift*1000)+' -f0shift '+str(f0shift*1000)+' -f0file '+tmpf0file+' -fftl '+str(dftlen)+' -apord '+str(dftlen/2+1)+' -raw '+tmprawwavfile+' '+tmpaperfile # -nmsg
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

    tmprawwavfile = sigproc.gentmpfile('straight-analysis.raw')
    tmpf0file = sigproc.gentmpfile('straight-analysis.f0')
    tmpspecfile = sigproc.gentmpfile('straight-analysis.spec')
    tmpaperfile = sigproc.gentmpfile('straight-analysis.aper')

    try:
        ((wav.copy()*np.iinfo(np.int16).max).astype(np.int16)).tofile(tmprawwavfile)

        np.savetxt(tmpf0file, f0s[:,1], fmt='%f')

        #print('STRAIGHTPATH='+STRAIGHTPATH)

        # Aperiodicity estimation
        cmd = STRAIGHTPATH+'straight_bndap -float -nmsg -f '+str(fs)+' -shift '+str(shift*1000)+' -f0shift '+str(f0shift*1000)+' -f0file '+tmpf0file+' -fftl '+str(dftlen)+' -apord '+str(dftlen/2+1)+' -raw '+tmprawwavfile+' '+tmpaperfile # -nmsg
        os.system(cmd)

        # Spectral envelope estimation
        cmd = STRAIGHTPATH+'straight_mcep -float -nmsg -f '+str(fs)+' -fftl '+str(dftlen)+' -apord '+str(dftlen/2+1)+' -shift '+str(shift*1000)+' -f0shift '+str(f0shift*1000)+' -f0file '+tmpf0file+' -pow -raw '+tmprawwavfile+' '+tmpspecfile # -nmsg
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

    if f0s.shape[0]!=SPEC.shape[0] or (APER!=None and (SPEC.shape[0]!=APER.shape[0])):
        raise ValueError('Features dont have the same length (F0={}, SPEC={}, APER={}).'.format(f0s.shape, SPEC.shape, APER.shape))
    if APER!=None and (SPEC.shape[1]!=APER.shape[1]):
        raise ValueError('Spectral features dont have the same width (SPEC={}, APER={}).'.format(SPEC.shape, APER.shape))

    dftlen = (SPEC.shape[1]-1)*2

    if APER==None:
        APER = np.zeros(SPEC.shape)
        APER[f0s[:,1]>0,:int((float(dftlen)/fs)*4000.0)] = -100.0

    if f0_forcecont:
        # Replace zero values by interpolated values
        idx = f0s[:,1]>0
        f0s[:,1] = np.exp(np.interp(f0s[:,0], f0s[idx,0], np.log(f0s[idx,1])))

    tmpf0file = sigproc.gentmpfile('straight-synthesis.f0')
    tmpaperfile = sigproc.gentmpfile('straight-synthesis.aper')
    tmpspecfile = sigproc.gentmpfile('straight-synthesis.spec')
    tmpwavfile = sigproc.gentmpfile('straight-synthesis.wav')

    try:
        np.savetxt(tmpf0file, f0s[:,1], fmt='%f')
        APER.astype(np.float32).tofile(tmpaperfile)
        SPEC.astype(np.float32).tofile(tmpspecfile)

        cmd = STRAIGHTPATH+'synthesis_fft \
            -f '+str(fs)+' \
            -fftl '+str(dftlen)+' \
            -spec \
            -shift '+str(shift*1000)+' \
            -sigp 1.2 \
            -sd 0.5 \
            -cornf 4000 \
            -bw 70.0 \
            -delfrac 0.2 \
            -apfile '+tmpaperfile+' \
            -float \
            '+tmpf0file+' \
            '+tmpspecfile+' \
            '+tmpwavfile+' ' #>/dev/null 2>&1
        if verbose==0:
            cmd += ' > /dev/null 2>&1'
        #print(cmd)
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
