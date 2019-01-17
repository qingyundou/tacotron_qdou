# cython: language_level=3
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

from __future__ import division

cimport cython
import numpy as np
cimport numpy as np

#from libc.math cimport cos
#from libc.math cimport sin
from libc.math cimport log
from libc.math cimport sqrt

import time
import sys
from scipy.io import wavfile
from scipy import signal as sig
import pickle as pkl
import sigproc as sp

DTYPE = np.float64
CDTYPE = np.complex128
ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t CDTYPE_t

def maxnbh(sins):
    h = 0;
    for n in range(len(sins)):
        h = np.max((h, np.shape(sins[n])[1]))
    return h

#@cython.boundscheck(False)
#@cython.wraparound(False) # [-1] doesn't work anymore!
#@cython.nonecheck(False)
#def wrap(_p):
    #cdef int idx
    #cdef DTYPE_t v
    #cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] p = np.array(_p, dtype=DTYPE)

    #for idx in range(len(p)):
        #v = p[idx]
        #v -= int(v/(2*np.pi))*(2*np.pi)
        #if v>np.pi:
            #v -= 2*np.pi
        #elif v<-np.pi:
            #v += 2*np.pi
        #p[idx] = v

    #return p
    ##return np.arctan2(np.sin(p), np.cos(p))
    #return np.angle(np.exp(1j*_p))


@cython.boundscheck(False)
#@cython.wraparound(False) # [-1] doesn't work anymore!
@cython.nonecheck(False)
@cython.cdivision(True)
def extract_peaks(CDTYPE_t[:] S, DTYPE_t fs, DTYPE_t f0, int winlen, int dftlen, quadraticfit=True):
    cdef DTYPE_t[:] SF
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] SA, SP
    cdef DTYPE_t f, ca, la, ra, A, B, C, dx, imaxv, tmpf
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] freqs, amps, phases

    # The initial frequencies are
    freqs = f0 * np.arange(int(np.floor((fs/2.0-f0/2.0)/f0))+1)

    SA = np.abs(S)
    SA[SA==0.0] = np.finfo(SA[0]).eps
    SA = np.log(SA)
    # SA[np.isinf(SA)] = np.finfo(SA[0]).min    # This one prevent the zeros but let a warning triggered.

    SF = (float(fs)/dftlen)*np.arange(dftlen/2+1)

    # Keep track of "peakiness" of the values
    issins = np.ones(len(freqs), dtype=bool)

    # Estimate the amplitudes from the clostest peaks to the initial freqs
    amps = np.zeros(len(freqs), dtype=DTYPE)
    binshalfrange = int(np.floor((f0*dftlen/fs)/2))
    binsrangeidx = np.arange(-binshalfrange,binshalfrange+1, dtype=np.int64)
    for fi in range(len(freqs)):
        #print("fi="+str(fi))
        f = freqs[fi]
        if fi==0:
            amps[0] = SA[0]
            issins[fi] = SA[1]<SA[0]
            continue

        # Catch the peak in the search range
        cenb = int(f*dftlen/fs+0.5) # 0.5 for proper rounding

        if not quadraticfit:
            cenb = np.clip(cenb, 0, dftlen/2)
            amps[fi] = SA[cenb]
            if cenb<dftlen/2:
                issins[fi] = SA[cenb-1]<SA[cenb] and SA[cenb]>SA[cenb+1]
        else:
            # Find all the peaks in the search range
            pidx = np.where(np.diff(np.sign(np.diff(SA[cenb+binsrangeidx])))<0)[0]
            if len(pidx)==0:
                issins[fi] = False
                continue
            issins[fi] = True

            # Take the closest to the expected one
            pidxc = np.argmin((pidx+1)-binshalfrange)
            aimax = cenb+binsrangeidx[pidx[pidxc]+1]
            ca = SA[aimax]
            la = SA[aimax-1]
            ra = SA[aimax+1]

            # If a peak has been found, do parabolic fitting
            A = (ra-ca + la-ca)/2.0
            B = -la+ca+A
            C = la-A+B

            # Position of the peak in relative bins
            dx = -B/(2.0*A)

            # Correct the frequency ...
            freqs[fi] = (aimax+dx)*fs/dftlen
            # ... and set the amplitude
            amps[fi] = A*dx*dx + B*dx + C

    # Estimate the amplitude of non-peaks
    amps[~issins] = np.interp(freqs[~issins], SF, SA)
    amps = np.exp(amps)

    # Estimate the phase by linear interpolation
    SP = np.angle(S)
    # Move the phase to the window's center
    SP += (((winlen-1)/2)*2*np.pi/float(dftlen))*np.arange(int(dftlen/2)+1)
    phases = sp.wrap(np.interp(freqs, SF, np.unwrap(SP)))

    return np.row_stack((freqs, amps, phases, issins))


@cython.boundscheck(False)
#@cython.wraparound(False) # [-1] doesn't work anymore!
@cython.nonecheck(False)
@cython.cdivision(True)
def compute_dft(wav4win, DTYPE_t fs, np.ndarray[DTYPE_t, ndim=1, mode="c"] win, int dftlen, np.ndarray[np.int64_t, ndim=1, mode="c"] winidx=None, ams=None, fms=None, fmpoly=1):

    winlen = len(win)
    wav4win = wav4win.copy()
    ci = (winlen-1)/2

    # If AM modulation is provided, demodulate
    if ams!=None:
        winam = ams[winidx]
        winam /= winam[ci]
        #winam *= len(winam)/np.sqrt(np.sum(winam**2)) # Preserve energy
        wav4win /= winam

    # If FM modulation is ...
    if fms==None:
        # ... not provided, use the traditional FFT
        S = np.fft.rfft(win*wav4win, dftlen)
    else:
        # ... provided, demodulate
        winfm = fms[winidx]

        y = win*wav4win

        # Frequency bins
        ks = np.arange(dftlen/2+1)
        # Time samples
        nn = np.arange(winlen) - (winlen-1)/2.0 # Do not remove! Necessary for the slope construction

        afull = winfm/winfm[ci]

        if fmpoly==1:
            # Use FChT
            pollin = np.polyfit(np.arange(winlen), afull, 1)
            ahat = pollin[0] # FChT's slope

            if ahat>2.0/winlen:  ahat = 2.0/winlen
            elif ahat<-2.0/winlen:  ahat = -2.0/winlen

            E = (-2*np.pi/dftlen)*np.matrix(ks).T*((1+0.5*ahat*nn)*nn)
            E = np.exp(1j*E)
            X = np.sum(np.multiply(np.ones((dftlen/2+1,1))*(y*np.sqrt(np.abs(1+ahat*nn))),E), axis=1)

        else:
            # Use aDFT
            # TODO Normalization factor (np.sqrt(np.abs(1+ahat*nn)) for FChT)

            if fmpoly>1:
                polyc = np.polyfit(np.arange(winlen), afull, fmpoly)
                afull = np.polyval(polyc, np.arange(winlen))

            E = (-2*np.pi/dftlen)*np.matrix(ks).T*(afull*nn)
            E = np.exp(1j*E)
            X = np.sum(np.multiply(np.ones((dftlen/2+1,1))*y,E), axis=1)

        S = np.zeros((dftlen/2+1), dtype=np.complex128)
        for k in range(dftlen/2+1): S[k] = X[k,0]
        # Put the window's delay, to follow DFT convention
        S *= np.exp(-1j*(((winlen-1)/2)*2*np.pi/float(dftlen))*np.arange(int(dftlen/2)+1))

        #if winidx[0]/float(fs)>0.15 and 0:
            ##plt.plot(np.arange(winlen), winfm, 'k')
            ##plt.plot(np.arange(winlen), np.polyval(pollin, np.arange(winlen)), 'b')

            ##plt.plot(x, 'k')
            ##plt.plot(winam, 'b')
            ##plt.plot(y, 'r')
            #plt.plot(mag2db(np.fft.rfft(win*wav4win, dftlen)), 'k')
            #plt.plot(mag2db(S), 'b')
            ##plt.plot(np.angle(np.fft.rfft(win*wav4win, dftlen)), 'k')
            ##plt.plot(np.angle(S), 'b')
            #from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return S


@cython.boundscheck(False)
#@cython.wraparound(False) # [-1] doesn't work anymore!
@cython.nonecheck(False)
@cython.cdivision(True)
def estimate_sinusoidal_params(np.ndarray[DTYPE_t, ndim=1] wav, DTYPE_t fs, np.ndarray[DTYPE_t, ndim=2] f0s, DTYPE_t nbper=3, ams=None, fms=None, dropoutwins=False, quadraticfit=True, verbose=1):
    cdef int fi, winlen, dftlen, binshalfrange, cenb, imax, aimax, argi
    cdef DTYPE_t f, f0, ca, la, ra, A, B, C, dx, imaxv, tmpf
    #cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] freqs, amps, phases
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] winidx #, binsrangeidx
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] win
    cdef CDTYPE_t[:] S
    #start = time.time()

    # Replace 0 by interpolations
    f0s = f0s.copy()
    f0s[:,1] = np.interp(f0s[:,0], f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])

    #print("Sinusoidal analysis: f0 in ["+str(np.min(f0s[:,1]))+","+str(np.max(f0s[:,1]))+"]")

    #sins = [None]*len(f0s[:,0])
    sins = []
    winisins = [True]*len(f0s[:,0])
    for n, t in enumerate(f0s[:,0]):
        f0 = f0s[n,1]
        if verbose>0:
            print("\rSinusoidal analysis (cython version) t={:0.3f}s({:.0f}%) f0={:0.2f}Hz".format(t, 100*t/f0s[f0s.shape[0]-1,0], f0))

        #if t<0.35: continue
        #if t<0.837: continue

        # Window's length
        winlen = int(0.5+(nbper*fs/f0)/2)*2+1 # with rounding

        # Extract the signal segment to analyse
        winidx = np.arange(-int((winlen-1)/2),int((winlen-1)/2+1), dtype=np.int64)
        winidx += int(0.5+fs*t)
        if winidx[0]<0 or winidx[-1]>=len(wav):
            winisins[n] = False
            if dropoutwins:
                continue
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
            sin = np.row_stack((freqs, sys.float_info.epsilon*np.ones(np.shape(freqs)), sp.wrap(2*np.pi*np.random.rand(len(freqs))), np.zeros(np.shape(freqs), dtype=bool)))
            sins.append(sin)
            continue

        # Window's shape
        win = np.blackman(winlen)
        win = win/sum(win) # Normalize for sinusoidal content

        # Compute the DFT
        dftlen = 2**(int(np.log2(winlen)+1)+1)

        S = compute_dft(wav4win, fs, win, dftlen, winidx, ams, fms)

        sin = extract_peaks(S, fs, f0, winlen, dftlen, quadraticfit=quadraticfit)
        sins.append(sin)

        if 0:
            #SF = (float(fs)/dftlen)*np.arange(dftlen/2+1)
            #SA = np.log(np.abs(S))
            #SA[np.isinf(SA)] = np.finfo(SA[0]).min
            plt.plot(SF, SA, 'k')
            plt.plot(freqs, np.log(amps), 'xk')
            print(n)
            print(sins[n])
            from IPython.core.debugger import  Pdb; Pdb().set_trace()

    if verbose>0:
        print('\r                                                               \r')

    #end = time.time()
    #print("\nElapsed: "+str(end - start))

    if dropoutwins:
        idx = np.where(winisins)[0]
        f0s = f0s[idx,:]

    return sins, f0s


@cython.boundscheck(False)
#@cython.wraparound(False) # [-1] doesn't work anymore!
@cython.nonecheck(False)
@cython.cdivision(True)
def synthesize_harmonics(f0s, sins, DTYPE_t fs, int wavlen, syndc=True, usephase=True, defampdb=-300):

    #cdef int fi
    cdef DTYPE_t defamp = np.log(sp.db2mag(defampdb))
    cdef int T = len(f0s[:,0])
    cdef int H, h, n
    #cdef np.ndarray[np.int64_t, ndim=1, mode="c"] idx
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] wav = np.zeros(wavlen)
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] times = np.array(range(wavlen))/float(fs)
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] dcat, f1, p1, p1at, ah, ph, am, pm

    # Synthesize the DC
    if syndc:
        dcat = np.zeros(T)
        for n in range(T) :
            dcat[n] = np.sign(np.cos(sins[n][2,0]))*sins[n][1,0]
        wav += np.interp(times, f0s[:,0], dcat)

    # Prepare the fundamental phase
    f1 = np.interp(times, f0s[:,0], f0s[:,1])
    p1 = (2*np.pi/fs)*np.cumsum(f1)
    p1at = np.interp(f0s[:,0], times, p1)

    if T==1: # If there is only one value...
        p1 -= p1at[0] # ...remove it from the p1 already

    # Harmonic synthesis
    H = maxnbh(sins)
    for h in range(1,H):
        #print("\rh="+str(h)+"                                          ")

        if T==1:
            # Add only where frequency < nyquist
            wav += 2*sins[n][1,h]*np.cos(h*p1+sins[n][2,h])
        else:
            # Extract the amplitude and phase for each anchor time
            ah = defamp*np.ones(T)
            ph = sp.wrap(2*np.pi*np.random.rand(T))
            for n in range(T) :
                if h>=np.shape(sins[n])[1]:
                    continue
                ah[n] = np.log(sins[n][1,h])
                if usephase:
                    ph[n] = sins[n][2,h] - h*p1at[n]

            # Switch to continuous
            am = np.exp(np.interp(times, f0s[:,0], ah))
            pm = h*p1
            if usephase:
                pm += np.interp(times, f0s[:,0], np.unwrap(ph))

            # Add only where frequency < nyquist
            idx = h*f1<fs/2.0
            wav[idx] = wav[idx] + 2*am[idx]*np.cos(pm[idx])

    #print('\r')

    return wav

def smooth_params(f0s, sins, fs, ampab=None, phaseab=None):

    wavlen = int(np.ceil(fs*f0s[-1,0]))
    times = np.arange(wavlen)/float(fs)

    T = len(f0s[:,0])

    # Prepare the fundamental phase
    f1 = np.interp(times, f0s[:,0], f0s[:,1])
    p1 = (2*np.pi/fs)*np.cumsum(f1)
    p1at = np.interp(f0s[:,0], times, p1)

    H = maxnbh(sins)
    ah = np.zeros(T)
    ph = np.zeros(T)
    for h in range(H):
        used = np.zeros(T)
        for n in range(T) :
            if h>=np.shape(sins[n])[1]: continue
            used[n] = True
            if ampab!=None:
                ah[n] = np.log(sins[n][1,h])
            if phaseab!=None:
                ph[n] = sins[n][2,h] - h*p1at[n]

        idx = np.where(used)[0]
        if ampab!=None:
            ah = np.interp(f0s[:,0], f0s[idx,0], ah[idx])
            ahf = sig.filtfilt(ampab[0], ampab[1], ah)
        if phaseab!=None:
            ph = np.unwrap(ph)
            ph = np.interp(f0s[:,0], f0s[idx,0], ph[idx])
            ph = np.unwrap(ph)
            phf = sig.filtfilt(phaseab[0], phaseab[1], ph)

        for n in range(T) :
            if h>=np.shape(sins[n])[1]: continue
            if ampab!=None:
                sins[n][1,h] = np.exp(ahf[n])
            if phaseab!=None:
                sins[n][2,h] = sp.wrap(phf[n] + h*p1at[n])

    return sins

@cython.boundscheck(False)
@cython.wraparound(False) # [-1] doesn't work anymore!
@cython.nonecheck(False)
@cython.cdivision(True)
def estimate_pdd(sins, f0s, DTYPE_t fs, int nbperperiod, int dftlen, np.ndarray[DTYPE_t, ndim=2] A=None, outPD=False, rmPDtrend=True, outFscale=True, extrapDC=False, outCircVar=False):
    '''
        A : [log|.|]
    '''

    if len(sins)!=f0s.shape[0]:
        raise ValueError('f0s and sinusoidal parameters need to have the same length.')

    cdef int H = maxnbh(sins)

    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] PD = np.zeros((len(sins), H), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] PDS = np.zeros((len(sins), H), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] PDD = np.zeros((len(sins), H), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] PDR = np.zeros((len(sins), H), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] PDSc, PDSs, PDDc, PDDs

    cdef DTYPE_t[:] F = (float(fs)/dftlen)*np.arange(dftlen/2+1)
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] PDDF = np.zeros((len(sins), dftlen/2+1), dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] P, vtfcc, RP, pdh
    cdef int n, N, k
    cdef DTYPE_t f0, v, v1, v2

    f0s = f0s.copy()
    f0s[:,1] = np.interp(f0s[:,0], f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])

    N = len(sins)
    cdef int vtf_dftlen = -1
    if A!=None:
        vtf_dftlen = (len(A[0,:])-1)*2

    #pdstart = time.time()

    for n in range(N):
        #f0 = sins[n][0,1] # Should use f0, not f1 !
        f0 = f0s[n,1]

        P = sins[n][2,:]
        P[0] = 0

        if A!=None:
            # Remove the VTF phase
            vtfcc = np.fft.irfft(A[n,:])
            vtfcc = vtfcc[:vtf_dftlen/2+1]
            vtfcc[1:] *= 2
            VTFP = np.imag(np.fft.rfft(vtfcc, vtf_dftlen))
            P -= np.interp(sins[n][0,:], F, VTFP)

        # Compute the Phase Distortion (PD)
        RP = np.arange(np.shape(sins[n])[1])*P[1] # Relative Phase Shift
        RP = P - RP
        pdh = np.diff(np.unwrap(RP))

        PD[n,0:len(pdh)] = np.unwrap(pdh)

    #pdend = time.time()
    #print("\nPD: Elapsed: "+str(pdend - pdstart))

    #pdsstart = time.time()

    if rmPDtrend:
        # Compute the smooth PD (the local trend)
        bcoefs = np.hamming(2*nbperperiod+1) # 9 From article
        bcoefs = bcoefs/sum(bcoefs)
        PDc = np.cos(PD)
        PDs = np.sin(PD)
        PDc = sig.medfilt(PDc, (nbperperiod*2+1, 1))
        PDs = sig.medfilt(PDs, (nbperperiod*2+1, 1))
        PDSc = sig.filtfilt(bcoefs, [1], PDc, axis=0)
        PDSs = sig.filtfilt(bcoefs, [1], PDs, axis=0)
        PDSs = PDSs.copy(order='C')
        PDSc = PDSc.copy(order='C')
        PDS = np.arctan2(PDSs,PDSc) # The smooth PD (the trend)
        PDR = PD - PDS
    else:
        PDR = PD

    #print("\nPDS: Elapsed: "+str(time.time() - pdsstart))
    #pddstart = time.time()

    # Compute the PDD
    bcoefs = np.ones(2*nbperperiod+1) # 9 From article
    bcoefs = bcoefs/sum(bcoefs)
    PDDc = sig.filtfilt(bcoefs, [1], np.cos(PDR), axis=0)
    PDDs = sig.filtfilt(bcoefs, [1], np.sin(PDR), axis=0)
    #fc = 25.0/4.0
    #shift = np.mean(np.diff(f0s[:,0]))
    #print(shift)
    #b, a = sig.butter(8, fc/(0.5/shift))
    #b, a = sig.butter(8, 1.0/32.0)
    #PDDc = sig.filtfilt(b, a, np.cos(PDR), axis=0)
    #PDDs = sig.filtfilt(b, a, np.sin(PDR), axis=0)
    #if not outFisherSTD: PDD[:,:] = 1.0
    for n in range(N):
        for k in range(H):
            v1 = PDDc[n,k]
            v2 = PDDs[n,k]
            v = v1*v1+v2*v2 # Drop sqrt ...
            if outCircVar:
                # Circular variance
                PDD[n,k] = 1.0-np.sqrt(v)   # ... and put the sqrt here
            else:
                # Circular standard-deviation
                if v<1:
                    PDD[n,k] = sqrt(-log(v)) # ... and drop 2*

    #print("\nPDD: Elapsed: "+str(time.time() - pddstart))

    if outFscale:
        # Switch from harm to freq scale
        for n in range(N):
            #PDDF[n,:] = np.interp(F, sins[n][0,1:], PDD[n,:len(sins[n][0,1:])])
            PDDF[n,:] = np.interp(F, 0.5*f0s[n,1]+sins[n][0,1:], PDD[n,:len(sins[n][0,1:])])
            if extrapDC:
                PDDF[n,0:int(dftlen*sins[n][0,2]/fs)] = PDDF[n,int(dftlen*sins[n][0,2]/fs)] # Use the 1st harm, not f0
            else:
                PDDF[n,0:int(dftlen*sins[n][0,1]/fs)] = 0 # Use the 1st harm, not f0

        if outPD:
            PDF = np.zeros((len(sins), dftlen/2+1), dtype=DTYPE)
            for n in range(N):
                #PDF[n,:] = sp.wrap(np.interp(F, sins[n][0,1:], np.unwrap(PD[n,:len(sins[n][0,1:])])))
                PDF[n,:] = sp.wrap(np.interp(F, 0.5*f0s[n,1]+sins[n][0,1:], np.unwrap(PD[n,:len(sins[n][0,1:])])))
                #PDMF[n,0:int(dftlen*sins[n][0,1]/fs)] = 0 # Use the 1st harm, not f0
            return PDDF, PDF
        else:
            return PDDF
    else:
        if outPD:
            return PDD, PD
        else:
            return PDD
