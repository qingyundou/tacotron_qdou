#!/usr/bin/python

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

import argparse
import numpy as np

# sys.path.append('/home/degottex/Research/CUED/Code')

if  __name__ == "__main__" :

    argpar = argparse.ArgumentParser()
    argpar.add_argument("intspecfile", default=None, help="Input spectrum file")
    argpar.add_argument("--dftlen", default=4096, type=int, help="DFT size for the input spectrum")
    argpar.add_argument("--fs", default=16000, type=int, help="Sampling frequency[Hz]")
    argpar.add_argument("--outdftlen", type=int, help="Number of bands in the warped spectral representation")
    argpar.add_argument("--outlog", action='store_true', help="Store output in log amplitude")
    argpar.add_argument("outspecfile", default=None, help="Output spectrum file")
    args, unknown = argpar.parse_known_args()

    SPEC = np.fromfile(args.intspecfile, dtype=np.float32)
    SPEC = SPEC.reshape((-1, int(args.dftlen/2)+1))
    RSPEC = np.zeros((SPEC.shape[0], int(args.outdftlen/2)+1))
    for n in xrange(SPEC.shape[0]):
        rcc = np.fft.irfft(np.log(abs(SPEC[n,:])))
        rcc = rcc[:(args.dftlen/2)+1] # Take the second half since it has to go away anyway
        rcc[1:-1] *= 2 # Compensate for the energy loss of cutting the second half
        rcc = rcc[:int(args.outdftlen/2)+1] # Cut according to the new size
        rcc[-1] *= 0.5  # This one is supposed to be half the energy of the other bins (not 100% sure of this TODO)
        RSPEC[n,:] = np.real(np.fft.rfft(rcc, args.outdftlen))
    if not args.outlog:
        RSPEC = np.exp(RSPEC)
    RSPEC.astype('float32').tofile(args.outspecfile)

    if 0:
        shift = 0.005
        import matplotlib.pyplot as plt
        plt.ion()
        ts = shift*np.arange(SPEC.shape[0])
        plt.subplot(211)
        plt.imshow(np.log(SPEC).T, origin='lower', aspect='auto', interpolation='none', cmap='jet', extent=[0.0, ts[-1], 0.0, args.fs/2])
        plt.subplot(212)
        if not args.outlog:
            RSPEC = np.log(RSPEC)
        plt.imshow(RSPEC.T, origin='lower', aspect='auto', interpolation='none', cmap='jet', extent=[0.0, ts[-1], 0.0, args.fs/2])
        from IPython.core.debugger import  Pdb; Pdb().set_trace()
