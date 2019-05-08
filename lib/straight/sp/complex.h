/* complex.h
 *	coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __COMPLEX_H
#define __COMPLEX_H

#include "vector.h"

typedef struct FCOMPLEX_STRUCT {
    long length;
    FVECTOR real;
    FVECTOR imag;
} *FCOMPLEX;

typedef struct DCOMPLEX_STRUCT {
    long length;
    DVECTOR real;
    DVECTOR imag;
} *DCOMPLEX;

extern FCOMPLEX xfcalloc(long length);
extern FCOMPLEX xfczeros(long length);
extern void xfcfree(FCOMPLEX cplx);
extern void fccopy(FCOMPLEX cplx, FVECTOR real, FVECTOR imag);
extern FCOMPLEX xfccreate(FVECTOR real, FVECTOR imag, long length);

extern DCOMPLEX xdcalloc(long length);
extern DCOMPLEX xdczeros(long length);
extern void xdcfree(DCOMPLEX cplx);
extern void dccopy(DCOMPLEX cplx, DVECTOR real, DVECTOR imag);
extern DCOMPLEX xdccreate(DVECTOR real, DVECTOR imag, long length);

extern DVECTOR xdcpower(DCOMPLEX cplx);
extern DVECTOR xdcabs(DCOMPLEX cplx);
extern DVECTOR xdvcabs(DVECTOR real, DVECTOR imag, long length);
extern DVECTOR xdvcpower(DVECTOR real, DVECTOR imag, long length);

extern DCOMPLEX xdvcexp(DVECTOR real, DVECTOR imag, long length);
extern void dcexp(DCOMPLEX cplx);
extern DCOMPLEX xdcexp(DCOMPLEX x);

#define xfcreal(cplx) xfvclone((cplx)->real)
#define xfcimag(cplx) xfvclone((cplx)->imag)
#define xdcreal(cplx) xdvclone((cplx)->real)
#define xdcimag(cplx) xdvclone((cplx)->imag)

#define xfcclone(cplx) xfccreate((cplx)->real, (cplx)->imag, (cplx)->length)
#define xdcclone(cplx) xdccreate((cplx)->real, (cplx)->imag, (cplx)->length)

#endif /* __COMPLEX_H */
