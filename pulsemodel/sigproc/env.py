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

import numpy as np
import sigproc as sp

# import matplotlib.pyplot as plt
# plt.ion()

def multi_linear(sins, fs, dftlen):
    A = np.zeros((len(sins), dftlen/2+1))
    F = (float(fs)/dftlen)*np.arange(dftlen/2+1)
    for n in range(len(sins)):
        #if n*0.005<1.966: continue
        #print('t='+str(n*0.005))
        A[n,:] = np.interp(F, sins[n][0,1:], np.log(sins[n][1,1:]))
        if 0:
            plt.plot(sp.log2db(A[n,:]))
            from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return A

# From COVAREP
def trueenv(lA, order, maxit=200, maxtol=1, usewin=True, presmooth_factor=None, debug=0):

    dftlen = (len(lA)-1)*2

    if not presmooth_factor is None:
        lPA = trueenv(lA, int(order/presmooth_factor), maxit=maxit, maxtol=maxtol, presmooth_factor=None, debug=0)
        slim = int(0.25*dftlen/order)
        lA[:slim] = np.real(lPA[:slim]) # Correct only the bins "around" the DC ("around" defined by the order)

    if usewin:
        order = int(np.round(1.2*order)) # [1] 1.66
        win = np.hamming(2*order+1)
        win = win[(len(win)-1)/2:]

    lA = np.real(lA)

    lA0 = lA.copy()

    n = 0 # nb iterations
    maxdiff = np.inf

    lV = lA
    cc = np.zeros(1+order)
    while n<maxit and maxdiff>maxtol:
        #print('iter: '+str(n))

        ccp = np.fft.irfft(lA)
        ccp = ccp[:dftlen/2+1]
        ccp[1:-1] *= 2

        if usewin:
            ccd = ccp
            ccd[:1+order] -= cc
            Ei = np.sqrt(np.sum(ccd[:1+order]**2))
            Eo = np.sqrt(np.sum(ccd[1+order:]**2))
            #Eo = np.sqrt(np.sum((cca[1+order:])**2))
            lamb = np.sqrt((Ei+Eo)/Ei)
            cc = lamb*ccd[:1+order] + cc # Eq. (5) in [1] # TODO Doesn't work !?!?!
            #lamb = (Ei+Eo)/Ei
            #cc = (lamb*win)*ccd[:1+order] + cc # Eq. (5) in [1] # TODO Doesn't work !?!?!
            #print('cc ener='+str(np.sqrt(np.sum(cc**2))))
            #cc = lamb*win*ccp
        else:
            cc = ccp

        lV = np.fft.rfft(cc, dftlen)
        lV = np.real(lV)

        lA = np.maximum(lA,lV)    # Max of log amplitudes

        maxdiff = np.max(lA0-lV) # Can create over-shot
        #print('maxdiff='+str(maxdiff))

        if debug>0:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(lA0, 'k')
            #plt.plot(np.fft.rfft(ccp, dftlen), 'g')
            plt.plot(lV, 'r')
            plt.plot(lA, 'b')
            #plt.ylim((-55, -48))
            plt.ylim((-0.02, 0.001))
            plt.xlim((0.0, dftlen/2))
            from IPython.core.debugger import  Pdb; Pdb().set_trace()

        n += 1

    #print('nb iter={}, maxdiff={}'.format(n,maxdiff))

    if debug>0:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.ion()
        plt.plot(lA0, 'k')
        plt.plot(lA, 'b')
        plt.ylim((-10.0, 1))
        from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return lV
