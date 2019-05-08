#ifndef __VOICE_STRENGTH_H
#define __VOICE_STRENGTH_H

extern DMATRIX voice_strength(DVECTOR x, DVECTOR f0l, double fs, double framem,
			      double shiftm, long fftl, double eta);

#endif /* __VOICE_STRENGTH_H */
