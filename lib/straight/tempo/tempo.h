/* tempo.h
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __TEMPO_H
#define __TEMPO_H

extern DVECTORS tempo(DVECTOR x, double maxf0, double minf0, double f0ceil,
		      double f0floor, double fs, double shiftm,
		      double f0shiftm, char *cf0file, XBOOL f0var_flag,
		      XBOOL allparam_flag, XBOOL msg_flag, XBOOL onlycf0_flag);
extern DMATRICES fixpF0VexMltpBG4(DVECTOR x, double fs, double f0floor,
				 long nvc, long nvo, double mu,
				  double shiftm, double smp, double minm,
				 double pc, int nc, XBOOL allparam_flag);

#endif /* __TEMPO_H */
