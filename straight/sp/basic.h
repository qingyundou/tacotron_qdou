/* basic.h
 *	coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __BASIC_H
#define __BASIC_H

#define MAX_LONG 2147483647

extern void sp_warn_on(void);
extern void sp_warn_off(void);

/*extern double randun(void);
extern double gauss(double mu, double sigma);*/
extern double round(double x);
extern double fix(double x);
extern double rem(double x, double y);
extern void cexpf(float *xr, float *xi);
extern void cexp(double *xr, double *xi);
extern void clogf(float *xr, float *xi);
extern void clog(double *xr, double *xi);
extern void ftos(char *buf, double x);
extern void randsort(void *data, int num, int size);
extern long factorial(int n);
extern void decibel(double *x, long length);
extern void decibelp(double *x, long length);

double simple_random();
double simple_gnoise(double rms);

/*#define randn()	gauss(0.0, 1.0)*/
#define randn()	simple_gnoise(1.0)

#endif /* __BASIC_H */
