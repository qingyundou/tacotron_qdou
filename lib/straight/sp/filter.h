/* filter.h
 *	coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __FILTER_H
#define __FILTER_H

extern void dvangle(DVECTOR x);
extern DVECTOR xdvangle(DVECTOR x);
extern void dvunwrap(DVECTOR phs, double cutoff);
extern double sinc(double x, double c);
extern DVECTOR xdvlowpass(double cutoff, double sidelobe, double trans, double gain);
extern DVECTOR xdvfftfilt(DVECTOR b, DVECTOR x, long fftp);
extern DVECTOR xdvfftfiltm(DVECTOR b, DVECTOR x, long fftp);
extern DVECTOR xdvconv(DVECTOR a, DVECTOR b);
extern DVECTOR xdvfilter(DVECTOR b, DVECTOR a, DVECTOR x);

#endif /* __FILTER_H */
