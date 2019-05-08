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
import re
import numpy as np

import scipy.signal

import sigproc
from . import fileio, misc, resampling

def worldvocoder_is_available():
    import imp
    try:
        imp.find_module('pyworld')
        return True
    except ImportError:
        return False

def sv56demo_is_available():
    return misc.check_executable('sv56demo')

def sv56demo(wav, fs, level=-26):
    '''
    Normalize the waveform in amplitude
    '''
    misc.check_executable('sv56demo', 'source?')

    tmpinfname = sigproc.gentmpfile('interface-sv56demo-in.raw')
    tmpoutfname = sigproc.gentmpfile('interface-sv56demo-out.raw')

    try:
        (wav.copy()*np.iinfo(np.int16).max).astype(np.int16).tofile(tmpinfname)

        levelstr = ''
        if level!=None:
            levelstr = ' -lev '+str(level)
        cmd = os.path.join(sigproc.BINPATH, 'sv56demo') + ' ' + levelstr + ' -sf ' + str(fs) + ' ' + tmpinfname + ' ' + tmpoutfname + ' 640'
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        #print(out)
        
        # Put outputs in a dictionary
        m = re.findall(r'\n?\s*(.*): .* ([-+]?[0-9]+\.?[0-9]*).*', out)
        meta = dict(m)
        meta.pop('Input file', None)
        for key in meta: meta[key] = float(meta[key])
        
        wav = np.fromfile(tmpoutfname, dtype=np.int16)
        wav = wav / float(np.iinfo(np.int16).max)

    except:
        if os.path.exists(tmpinfname): os.remove(tmpinfname)
        if os.path.exists(tmpoutfname): os.remove(tmpoutfname)
        raise

    if os.path.exists(tmpinfname): os.remove(tmpinfname)
    if os.path.exists(tmpoutfname): os.remove(tmpoutfname)

    return wav, meta


def reaper_is_available():
    return misc.check_executable('reaper')

def reaper(wav, fs, shift, f0min, f0max, rmsteps=False, outpm=False, prehp_cutoff=None):
    '''
    Estimate the f0

    prehp_cutoff: High-pass the signal before estimating f0 (def. None). Value is [Hz].

    Source:
        https://github.com/google/REAPER
    '''
    misc.check_executable('reaper', 'You can download the source from https://github.com/google/REAPER')

    if not prehp_cutoff is None:
        b, a = scipy.signal.butter(8, prehp_cutoff/(0.5/shift), btype='high')
        wav = scipy.signal.filtfilt(b, a, wav)

    tmpwavfile = sigproc.gentmpfile('interface-reaper-in.wav')
    tmpf0file = sigproc.gentmpfile('interface-reaper-out.f0')
    tmppmfile = sigproc.gentmpfile('interface-reaper-out.pm')

    try:
        fileio.wavwrite(tmpwavfile, wav, fs, np.int16)

        cmd = os.path.join(sigproc.BINPATH, 'reaper') + ' -i ' + tmpwavfile + ' -m ' + str(f0min) + ' -x ' + str(f0max) + ' -e ' + str(shift) + ' -a -f ' + tmpf0file + ' -p ' + tmppmfile
        os.system(cmd+' 2>&1 >/dev/null') # Run it silent

        # Read the f0 values and the corresponding analysis times
        f0s = list()
        with open(tmpf0file) as f0file:
            n = 1
            for line in f0file:
                if n>7: # Skip the EST-file header TODO Neater way to do this!
                    values = line.split()
                    f0s.append((float(values[0]), float(values[2])))
                n += 1
        f0s = np.array(f0s)
        f0s[f0s[:,1]<0,1] = 0

        if rmsteps:
            f0s = resampling.f0s_rmsteps(f0s)

        # Read the f0 values and the corresponding analysis times
        pms = list()
        with open(tmppmfile) as pmfile:
            n = 1
            for line in pmfile:
                if n>7: # Skip the EST-file header TODO Neater way to do this!
                    values = line.split()
                    pms.append(float(values[0]))
                n += 1
        pms = np.array(pms)

    except: 
        if os.path.exists(tmpwavfile): os.remove(tmpwavfile)
        if os.path.exists(tmpf0file): os.remove(tmpf0file)
        if os.path.exists(tmppmfile): os.remove(tmppmfile)
        raise

    if os.path.exists(tmpwavfile): os.remove(tmpwavfile)
    if os.path.exists(tmpf0file): os.remove(tmpf0file)
    if os.path.exists(tmppmfile): os.remove(tmppmfile)

    if outpm:
        return f0s, pms
    else:
        return f0s
