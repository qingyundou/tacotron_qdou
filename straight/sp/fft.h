/* fft.h
 *	coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __FFT_H
#define __FFT_H

#include "vector.h"
#include "complex.h"

extern DCOMPLEX xvfft(DVECTOR real, DVECTOR imag, long length, int inv);
extern void cfft(DCOMPLEX cplx, int inv);
extern void cfftturn(DCOMPLEX cplx);
extern void cfftshift(DCOMPLEX cplx);

extern FVECTOR xfvfftconv(FVECTOR a, FVECTOR b, long fftl);
extern DVECTOR xdvfftconv(DVECTOR a, DVECTOR b, long fftl);

extern FVECTOR xfvfftpower(FVECTOR x, long fftl);
extern DVECTOR xdvfftpower(DVECTOR x, long fftl);
extern FVECTOR xfvfftabs(FVECTOR x, long fftl);
extern DVECTOR xdvfftabs(DVECTOR x, long fftl);
extern DVECTOR xdvfftangle(DVECTOR x, long fftl);
extern DVECTOR xdvfftgrpdly(DVECTOR x, long fftl);

extern void dvspectocep(DVECTOR x);
extern void dvceptospec(DVECTOR x);
extern DVECTOR xdvrceps(DVECTOR x, long fftl);
extern void dvlif(DVECTOR cep, long fftl, long lif);
extern void dvceptompc(DVECTOR cep);
extern DVECTOR xdvmpceps(DVECTOR x, long fftl);

extern DVECTOR xdvcspec(DVECTOR mag, DVECTOR phs);

extern void fvfftshift(FVECTOR x);
extern void dvfftshift(DVECTOR x);
extern void fvfftturn(FVECTOR x);
extern void dvfftturn(DVECTOR x);

extern void fvfft(FVECTOR x);
extern void dvfft(DVECTOR x);
extern void fvifft(FVECTOR x);
extern void dvifft(DVECTOR x);

extern FVECTOR xfvfft(FVECTOR x, long length);
extern DVECTOR xdvfft(DVECTOR x, long length);
extern FVECTOR xfvifft(FVECTOR x, long length);
extern DVECTOR xdvifft(DVECTOR x, long length);

extern int rfftabs(double *x, long fftp);
extern int rfftpow(double *x, long fftp);
extern int rfftangle(double *x, long fftp);

extern int nextpow2(long n);
extern void fftturnf(float *xRe, float *xIm, long fftp);
extern void fftturn(double *xRe, double *xIm, long fftp);
extern void fftshiftf(float *xRe, float *xIm, long fftp);
extern void fftshift(double *xRe, double *xIm, long fftp);
extern int fftf(float *xRe, float *xIm, long fftp, int inv);
extern int fft(double *xRe, double *xIm, long fftp, int inv);

#endif /* __FFT_H */
