/* kaiser.h
 *	coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __KAISER_H
#define __KAISER_H

extern void getkaiserparam(double sidelobe, double trans, double *beta, long *length);
extern int kaiser(double w[], long n, double beta);
extern double ai0(double x);

#endif /* __KAISER_H */
