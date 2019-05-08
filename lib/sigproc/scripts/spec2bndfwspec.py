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

import sys
import argparse
import numpy as np

sys.path.append('/home/degottex/Research/CUED/Code')
from lib import sigproc as sp

if  __name__ == "__main__" :

    argpar = argparse.ArgumentParser()
    argpar.add_argument("specfile", default=None, help="Input spectrum file")
    argpar.add_argument("--dftlen", default=4096, type=int, help="DFT size for the input spectrum")
    argpar.add_argument("--fs", default=16000, type=int, help="Sampling frequency[Hz]")
    argpar.add_argument("--nbbands", type=int, help="Number of bands in the warped spectral representation")
    argpar.add_argument("bndfwspecfile", default=None, help="Output frequency warped spectrum file")
    args, unknown = argpar.parse_known_args()

    SPEC = np.fromfile(args.specfile, dtype=np.float32)
    SPEC = SPEC.reshape((-1, int(args.dftlen / 2)+1))
    FWSPEC = sp.linbnd2fwbnd(np.log(SPEC), args.fs, args.dftlen, args.nbbands)
    FWSPEC.astype('float32').tofile(args.bndfwspecfile)

    if 0:
        shift = 0.005
        SPECR = np.exp(sp.fwbnd2linbnd(FWSPEC, args.fs, args.dftlen))
        import matplotlib.pyplot as plt
        plt.ion()
        ts = shift*np.arange(SPEC.shape[0])
        plt.subplot(211)
        plt.imshow(sp.mag2db(SPEC).T, origin='lower', aspect='auto', interpolation='none', cmap='jet', extent=[0.0, ts[-1], 0.0, args.fs/2])
        plt.subplot(212)
        plt.imshow(sp.mag2db(SPECR).T, origin='lower', aspect='auto', interpolation='none', cmap='jet', extent=[0.0, ts[-1], 0.0, args.fs/2])
        from IPython.core.debugger import  Pdb; Pdb().set_trace()
