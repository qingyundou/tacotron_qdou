/* window.h
 *	coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __WINDOW_H
#define __WINDOW_H

#include "vector.h"

extern void blackmanf(float window[], long length);
extern void blackman(double window[], long length);
extern FVECTOR xfvblackman(long length);
extern DVECTOR xdvblackman(long length);

extern void hammingf(float window[], long length);
extern void hamming(double window[], long length);
extern FVECTOR xfvhamming(long length);
extern DVECTOR xdvhamming(long length);

extern void hanningf(float window[], long length);
extern void hanning(double window[], long length);
extern FVECTOR xfvhanning(long length);
extern DVECTOR xdvhanning(long length);

extern void nblackmanf(float window[], long length);
extern void nblackman(double window[], long length);
extern FVECTOR xfvnblackman(long length);
extern DVECTOR xdvnblackman(long length);

extern void nhammingf(float window[], long length);
extern void nhamming(double window[], long length);
extern FVECTOR xfvnhamming(long length);
extern DVECTOR xdvnhamming(long length);

extern void nhanningf(float window[], long length);
extern void nhanning(double window[], long length);
extern FVECTOR xfvnhanning(long length);
extern DVECTOR xdvnhanning(long length);

extern void nboxcarf(float window[], long length);
extern void nboxcar(double window[], long length);
extern FVECTOR xfvnboxcar(long length);
extern DVECTOR xdvnboxcar(long length);

#endif /* __WINDOW_H */
