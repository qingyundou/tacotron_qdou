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
import sigproc as sp

if  __name__ == "__main__" :

    argpar = argparse.ArgumentParser()
    argpar.add_argument("bndspecfile", default=None, help="Input spectrum file")
    argpar.add_argument("--nbbands", type=int, help="Number of bands in the warped spectral representation")
    argpar.add_argument("--dftlen", default=4096, type=int, help="DFT size for the output spectrum")
    argpar.add_argument("--fs", default=16000, type=int, help="Sampling frequency[Hz]")
    argpar.add_argument("specfile", default=None, help="Output warped spectrum file")
    args, unknown = argpar.parse_known_args()

    BNDSPEC = np.fromfile(args.bndspecfile, dtype=np.float32)
    BNDSPEC = BNDSPEC.reshape((-1, args.nbbands))
    SPEC = np.exp(sp.fwbnd2linbnd(BNDSPEC, args.fs, args.dftlen))
    SPEC.astype('float32').tofile(args.specfile)
