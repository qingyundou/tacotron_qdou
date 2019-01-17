/* straight_body_sub.h
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __STRAIGHT_BODY_SUB_H
#define __STRAIGHT_BODY_SUB_H

extern double sb_extractf0(DVECTOR f0l, long idf0, double fguard,
			   double bguard, double f0shiftm);
extern DVECTOR sb_xwintable(long framel, double *refw);
extern DVECTOR sb_xgetpswin(DVECTOR wt, double t0, double refw);
extern DVECTOR sb_xcutsig(DVECTOR sig, long offset, long length);
extern DVECTOR sb_xgetfftpow(DVECTOR cx, DVECTOR wxe, long fftl,
			     double pc);
extern DVECTOR sb_xsmoothfreq(DVECTOR pw, double t0, long fftl);
extern DVECTOR sb_xgetsmoothwin(double t0);

/* for version 2.0 */
extern DVECTOR sb_xgetsinglewin(double t0, long framel, double *cf);
extern DVECTOR sb_xgetdoublewin(double t0, long framel, double cf);
extern DVECTOR sb_xgetdbfftpow(DVECTOR pws, DVECTOR pwd, double pc);
extern DVECTOR sb_xgetfftangle(DVECTOR cx, DVECTOR wxe, long fftl);
extern DVECTOR sb_xgetfftgrpdly(DVECTOR cx, DVECTOR wxe, long fftl);
extern DVECTOR sb_xleveleqspec(DVECTOR pw, double t0, long fftl);
extern DVECTOR sb_xgetsmoothwin_s(double t0, long fftl);
extern DVECTOR sb_xsmoothfreq_s(DVECTOR pw, double t0, long fftl);
extern DVECTOR sb_xtimeexpand(DVECTOR cumfreq, LVECTOR tx, long ii, 
			      long nframe);
extern LVECTOR sb_xfindregion(DVECTOR txx, double value, int eqflag);

/* for version 3.0 */
extern DVECTOR sb_xgetsinglewin2(double t0, long framel, double coef);
extern DVECTOR sb_xgetdoublewin2(DVECTOR wxs, double t0, long framel, 
				 double coef);
extern DVECTOR sb_xgetsmoothwin_s2(double t0, long fftl);
extern void sb_halfrectspec(DVECTOR spw);
extern DVECTOR sb_xsmoothfreq_s2(DVECTOR pw, double t0, long fftl);
extern void sb_blendspec(DVECTOR spw, DVECTOR pw, double lambh, double lambl,
			 double fs, long fftl);
extern DVECTOR sb_xlowcutwin(double fs, double bw, double cornf, long fftl);
extern void sb_xfree_sub(void);

/* for straight_body_c03, V30k18 (matlab) */
extern DVECTORS butter(long n, double cutoff);
extern DVECTOR buttap(long n);
extern DVECTOR xdvfilt(DVECTOR b, DVECTOR a, DVECTOR x);
extern DVECTOR optimumsmoothing(double eta, double pc);
extern DVECTOR sb_xsmoothfreq_c03(DVECTOR pw, double t0, long fftl,
				  DVECTOR ovc);
extern void sb_xsmoothtime_c03(DMATRIX n2sgram, DVECTOR f0l, double lowestf0,
			       double shiftm);
extern DVECTOR sb_xtconstuv_c03(DMATRIX n2sgram, DVECTOR xh2, DVECTOR xhh,
				DVECTOR sumvec, double ttlv, double ttlv2,
				double fs, double shiftm, long ncw);
extern DVECTOR xgethannexpwin(double fs, long ncw);
extern DVECTOR xgetpowsig(DVECTOR x, DVECTOR w, double ttlv, double shiftl,
			  long ncw, long len);
extern void specreshape(DMATRIX n2sgram, DVECTOR f0l, double fs, double eta,
			double pc, double mag, XBOOL msg_flag);
extern DMATRIX aperiodicpart4(DVECTOR x, DVECTOR f0l, double fs,
			      double shiftm, double intshiftm, long mm,
			      XBOOL msg_flag);
extern DVECTOR interp(DVECTOR x, long r);
extern DVECTOR xdvwin_aperiodicpart4(long fftl, double fs, double fr40);
extern DVECTOR interp1l(DVECTOR x, DVECTOR y, DVECTOR xi);
extern LVECTOR ss_xqsortidx(double *real, long length);
extern void ss_quicksort(double *array, long lower, long upper, LVECTOR idx);
extern DVECTOR interp1lq(DVECTOR x, DVECTOR y, DVECTOR xi);
extern void bisearch(double *x, long fidx, long lidx, double a, long *len,
		     long *ansidx);
extern DVECTOR xdvread_cf0file(char *cf0file, double *f0shift);
extern DVECTOR xget_interp_f0(DVECTOR f0l, double f0shift, double shift);


#define UNVOICED_F0 160.0


#endif /* __STRAIGHT_BODY_SUB_H */
