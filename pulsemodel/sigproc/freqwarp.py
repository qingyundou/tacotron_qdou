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
import numpy as np

import sigproc as sp
import misc

def bark_alpha(fs):
    '''
    Returns the alpha parameter corresponding to a Bark scale.
    '''
    return 0.8517*np.sqrt(np.arctan(0.06583*fs/1000.0))-0.1916

def erb_alpha(fs):
    '''
    Returns the alpha parameter corresponding to an ERB scale.
    '''
    return 0.5941*np.sqrt(np.arctan(0.1418*fs/1000.0))+0.03237


def loghspec2fwcep(C, fs, order=-1):
    '''
    From https://github.com/covarep/covarep/blob/master/envelope/hspec2fwcep.m
    '''

    if len(C.shape)>1:
        FWCEP = np.zeros((C.shape[0], 1+order))
        for n in range(C.shape[0]):
            FWCEP[n,:] = loghspec2fwcep(C[n,:], fs, order)
        return FWCEP


    # The input C is assumed to be a half spectrum
    dftlen = (len(C)-1)*2

    if order==-1:
        order = dftlen/2

    # Compute the warping function
    freqlin = fs*np.arange(dftlen/2+1)/dftlen
    freqmel = 0.5*fs*sp.lin2mel(freqlin)/sp.lin2mel(0.5*fs)

    # Warp the spectrum
    env = np.interp(freqlin, freqmel, C)

    # Compute the cepstrum
    fwcep = np.fft.irfft(env)

    # Drop the negative quefrencies and compensate the loss of cepstral energy
    fwcep = fwcep[0:1+order]
    fwcep[1:] *= 2

    if 0:
        import matplotlib.pyplot as plt
        plt.ion()

        Cmel = fwcep2loghspec(fwcep, fs, dftlen)

        plt.plot(C, 'k')
        plt.plot(env, 'r')
        plt.plot(Cmel, 'b')
        from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return fwcep


def fwcep2loghspec(fwcep, fs, dftlen):
    '''
    From https://github.com/covarep/covarep/blob/master/envelope/fwcep2hspec.m
    '''

    if len(fwcep.shape)>1:
        C = np.zeros((fwcep.shape[0], dftlen/2+1))
        for n in range(C.shape[0]):
            C[n,:] = fwcep2loghspec(fwcep[n,:], fs, dftlen)
        return C

    # Compute the warping function
    freqlin = fs*np.arange(dftlen/2+1)/dftlen
    freqmel = 0.5*fs*sp.lin2mel(freqlin)/sp.lin2mel(0.5*fs)

    # Decode the spectrum from the cepstral coefficients
    Cwrapl = np.real(np.fft.rfft(fwcep, dftlen))

    # Unwarp the spectrum
    C = np.interp(freqmel, freqlin, Cwrapl)

    return C


def linbnd2fwbnd(X, fs, dftlen, nbbnds):
    '''
    Split the spectral data X into mel-bands
    '''

    # Compute the warping function
    freqlin = fs*np.arange(dftlen/2+1)/dftlen
    freqmel = 0.5*fs*sp.lin2mel(freqlin)/sp.lin2mel(0.5*fs)

    # Warp the spectrum
    Z = np.zeros((X.shape[0],nbbnds))
    for t in np.arange(X.shape[0]):
        # Frequency warp the spectral data
        Y = np.interp(freqlin, freqmel, X[t,:])
        # Split in multiple bands
        for k in np.arange(nbbnds):
            ids = int((float(k)/nbbnds)*(dftlen/2+1))
            ide = int((float(k+1)/nbbnds)*(dftlen/2+1))
            Z[t,k] = np.mean(Y[ids:ide])

    return Z

def freq2fwspecidx(freq, fs, nbbnds, dftlen=4096):
    '''
    Retrieve the closest index to a frequency in frequency-warped spectrum.
    '''
    # TODO That's a bit whatever, should just use sp.lin2mel, bcs linbnd2fwbnd is actually averaging values in bands
    FF = fs*np.arange(dftlen/2+1)/float(dftlen)
    fwFF = linbnd2fwbnd(FF[np.newaxis,:], fs, dftlen, nbbnds)
    return np.min(np.where(fwFF>freq)[1])

def fwbnd2linbnd(Z, fs, dftlen, smooth=False):
    '''
    Reconstruct spectral data from mel-bands
    '''

    nbbnds = Z.shape[1]

    # Compute the warping function
    freqlin = fs*np.arange(dftlen/2+1)/dftlen
    freqmel = 0.5*fs*sp.lin2mel(freqlin)/sp.lin2mel(0.5*fs)

    # Warp the spectrum
    X = np.zeros((Z.shape[0],dftlen/2+1))
    for t in np.arange(X.shape[0]):

        # Decode the mel spectral data from the bands
        for k in np.arange(nbbnds):
            ids = int((float(k)/nbbnds)*(dftlen/2+1))
            ide = int((float(k+1)/nbbnds)*(dftlen/2+1))
            X[t,ids:ide] = Z[t,k]

        if smooth:
            rcc = np.fft.irfft(X[t,:])
            rcc = rcc[:dftlen/2+1]
            rcc[1:] *= 2
            rcc = rcc[:nbbnds]
            X[t,:] = np.real(np.fft.rfft(rcc, dftlen))

        # Unwarp the spectral data
        X[t,:] = np.interp(freqmel, freqlin, X[t,:])

    return X


def spec2mcep(SPEC, alpha, order):
    misc.check_executable('mcep', 'You can find it in SPTK (https://sourceforge.net/projects/sp-tk/)')

    dftlen = 2*(SPEC.shape[1]-1)

    tmpspecfile = sp.gentmpfile('spec2mcep.spec')
    outspecfile = sp.gentmpfile('spec2mcep.mcep')

    try:
        SPEC.astype(np.float32).tofile(tmpspecfile)
        cmd = os.path.join(sp.BINPATH,'mcep')+' -a '+str(alpha)+' -m '+str(int(order))+' -l '+str(dftlen)+' -e 1.0E-8 -j 0 -f 0.0 -q 3 '+tmpspecfile+' > '+outspecfile
        ret = os.system(cmd)
        if ret>0: raise ValueError('ERROR during execution of mcep')
        MCEP = np.fromfile(outspecfile, dtype=np.float32)
        MCEP = MCEP.reshape((-1, 1+int(order)))
    except:
        if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
        if os.path.exists(outspecfile): os.remove(outspecfile)
        raise

    if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
    if os.path.exists(outspecfile): os.remove(outspecfile)

    return MCEP

def mcep2spec(MCEP, alpha, dftlen):
    misc.check_executable('mgc2sp', 'You can find it in SPTK (https://sourceforge.net/projects/sp-tk/)')

    order = MCEP.shape[1]-1

    tmpspecfile = sp.gentmpfile('mcep2spec.mcep')
    outspecfile = sp.gentmpfile('mcep2spec.spec')

    try:
        MCEP.astype(np.float32).tofile(tmpspecfile)
        cmd = os.path.join(sp.BINPATH,'mgc2sp')+' -a '+str(alpha)+' -g 0 -m '+str(int(order))+' -l '+str(dftlen)+' -o 2 '+tmpspecfile+' > '+outspecfile
        ret = os.system(cmd)
        if ret>0: raise ValueError('ERROR during execution of mgc2sp')
        SPEC = np.fromfile(outspecfile, dtype=np.float32)
        SPEC = SPEC.reshape((-1, dftlen/2+1))
    except:
        if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
        if os.path.exists(outspecfile): os.remove(outspecfile)
        raise

    if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
    if os.path.exists(outspecfile): os.remove(outspecfile)

    return SPEC
