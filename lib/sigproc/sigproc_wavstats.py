'''
Copyright(C) 2017 Engineering Department, University of Cambridge, UK.

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
import argparse
import glob

import fileio

if  __name__ == "__main__" :
    argpar = argparse.ArgumentParser()
    argpar.add_argument("wavfiles", default=None, help="Input waveform files")
    args = argpar.parse_args()

    # First list the directories
    dirs = []
    if os.path.isdir(args.wavfiles):
        dirs.append(args.wavfiles)
        for indir in glob.glob(args.wavfiles+'/*'):
            if os.path.isdir(indir):
                dirs.append(indir)

    fileio.waveforms_statistics(dirs)
