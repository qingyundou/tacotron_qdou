/* straight_synth_sub.h
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __STRAIGHT_SYNTH_SUB_H
#define __STRAIGHT_SYNTH_SUB_H

#define SS_FRACTPHASE 1
#define SS_ALLVOICED 0
#define SS_NEGA_DIRECT 0

extern DVECTOR ss_xfractpitchtbl(long fftl);
extern DVECTOR ss_xlowcutwin(double fs, double cutf, double lowf0, long fftl);
extern DVECTOR ss_xphstranswin(long fftl);
extern DVECTOR ss_xgrpdlywin(double fs, double gdbw, long fftl);
extern DVECTOR ss_xfgrpdlywin(double fs, double gdbw, long fftl);
extern DVECTOR ss_xgdweight(double fs, double gdbw, double cornf, long fftl);
extern DVECTOR ss_xgetexactcep(DVECTOR pspec, DVECTOR cspec, double fract,
			       long fftl);
extern DVECTOR ss_xextractcep(DMATRIX n2sgram, double iix, LVECTOR ftbl, 
			      long fftl);
extern void ss_ceptompc(DVECTOR cep, long fftl);
extern double ss_getexactf0(double prevf0, double currf0, double nextf0,
			    double fract);
extern double ss_extractf0(DVECTOR f0l, double jjx);
extern DVECTOR ss_xgetuvwave(DVECTOR cep, DVECTOR lcutw, 
			     double period, long fftl);
extern DVECTOR ss_xfractpitchapf(double fract, long fftl);
extern DVECTOR ss_xgetgrpdly(DVECTOR fgdsw, DVECTOR gdwt, double fs, 
		      double gdbw, double gdsd, long fftl);
extern DVECTOR ss_xgdtorandomapf(DVECTOR gd, long fftl);
extern DVECTOR ss_xgetrandomapf(DVECTOR fgdsw, DVECTOR gdwt, double fs,
				double gdbw, double gdsd, long fftl);
extern DVECTOR ss_xgetwave(DVECTOR cep, DVECTOR lcutw, DVECTOR fgdsw, DVECTOR gdwt, 
			   double fract, double fs, double gdbw, double gdsd, 
			   long fftl, int rp_flag);
extern DVECTOR ss_xceptospec(DVECTOR cep, DVECTOR lcutw, long fftl);
extern DVECTOR ss_xspectowave(DVECTOR spc, long fftl);
extern DVECTOR ss_xceptowave(DVECTOR cep, DVECTOR apf, DVECTOR lcutw, long fftl);
extern void ss_fractpitchspec(DVECTOR spc, double fract, long fftl);
extern void ss_randomspec(DVECTOR spc, DVECTOR fgdsw, DVECTOR gdwt, double fs, 
			  double gdbw, double gdsd, long fftl);
extern DVECTOR ss_xspectouvwave(DVECTOR spc, double period, long fftl);
extern void ss_waveampcheck(DVECTOR wav);
extern void ss_xfree_sub(void);

/* function for V/UV mixing version */
extern void ss_getmixrate(double f0varh, double f0varl, 
			  double *mixh, double *mixl, int u_flag);
extern DVECTOR ss_xgetmixweight(LVECTOR ftbl, double mixh, double mixl, 
				double fs, long fftl);
extern DVECTOR ss_xceptospec_mix(DVECTOR cep, DVECTOR lcutw, DVECTOR mixwt, 
				 long fftl);
extern DVECTOR ss_xceptowave_mix(DVECTOR cep, DVECTOR apf, DVECTOR lcutw, 
				 DVECTOR mixwt, long fftl);

/* function for random allpass filter using f0 */
extern DVECTOR ss_xgetrandomapf_f0(double t0, double A, double B,
				   double a1, double a2, double p1, double
				   cornf, double gdbw, double fs, long fftl);
extern void ss_randomspec_f0(DVECTOR spc, double t0, double A, double B,
			     double a1, double a2, double p1, double cornf,
			     double gdbw, double fs, long fftl);

/* straight_synth_tb06	V30kr18 */
extern DVECTOR ss_xextractcep_tb06(DMATRIX n2sgram, long iix, LVECTOR ftbl,
				   long fftl, DVECTOR wlcut, double bias);
extern void ss_waveampcheck_tb06(DVECTOR x, double fs, double segms);
extern DMATRIX aperiodiccomp(DMATRIX dapv, double ashift, DVECTOR f0l,
			     double nshift);
extern DMATRIX aperiodiccomp_unit(DMATRIX dapv, double ashift, long *tinfo,
				  double nshift);
extern DVECTOR makeimaptm(DMATRIX n3sgram, double f0shiftm, double smflen,
			  double smfsht, double fs, double tsconv);
extern DVECTOR spcmovedb(DMATRIX n3sgram, double f0shiftm, double smflen,
			 double smfsht);



#endif /* __STRAIGHT_SYNTH_SUB_H */
