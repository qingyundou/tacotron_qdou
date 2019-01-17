import argparse
import numpy as np
import sys

import . straight

if  __name__ == "__main__" :
    argpar = argparse.ArgumentParser()
    argpar.add_argument("bapfile", help="Input bap file")
    argpar.add_argument("--bapsize", default=21, type=int, help="Size of a frame for the band aperiodicities")
    argpar.add_argument("--bapdtype", default='float32', help="The data type of the bap file")
    argpar.add_argument("--fs", default=16000.0, type=float, help="Sampling rate")
    argpar.add_argument("--sigp", default=1.2, type=float, help="Sigmoid parameter")
    argpar.add_argument("--aperdftlen", default=4096, type=int, help="Size of a frame for the aperiodicity")
    argpar.add_argument("--aperdtype", default='float32', help="The data format of the output aperiodicity")
    args = argpar.parse_args()

    # Calc the number of critical bands required for sampling frequency
    numbands = straight.nbbndap(args.fs)

    if args.bapsize!=None:
        numbands = args.bapsize
    #print(numbands)

    BAP = np.fromfile(args.bapfile, dtype=args.bapdtype)
    #print('BAP read size: '+str(BAP.shape[0]))
    BAP = BAP.reshape((-1, numbands))
    #print("BAP shape: "+str(BAP.shape))

    WNZ = straight.bndap2aper(BAP, args.fs, args.aperdftlen)

    if 0:
        import matplotlib.pyplot as plt
        plt.ion()
        f, axs = plt.subplots(2, 1, sharex=True, sharey=False)
        axs[0].imshow(BAP.T, origin='lower', aspect='auto', interpolation='none', vmin=-40, vmax=-5)
        axs[1].imshow(WNZ.T, origin='lower', aspect='auto', interpolation='none', vmin=-40, vmax=-5)
        from IPython.core.debugger import  Pdb; Pdb().set_trace()
        