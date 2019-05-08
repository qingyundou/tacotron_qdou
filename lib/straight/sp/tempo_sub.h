/* tempo_sub.h
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __TEMPO_SUB_H
#define __TEMPO_SUB_H

extern void cleaninglownoise(DVECTOR x, double fs, double f0fllor);
extern DVECTOR fir1(long length, double cutoff);
extern DVECTOR decimate(DVECTOR x, long dn);
extern DVECTORS cheby(long n, double rp, double cutoff);
extern DVECTOR chebap(long n, double rp, double *gain);
extern DVECTORS a2dbil(DVECTOR pv, double gain, double cutoff, XBOOL hp_flag);
extern DVECTOR xdvfiltfilt(DVECTOR b, DVECTOR a, DVECTOR x);
extern DMATRIX multanalytFineCSPB(DVECTOR x, double fs, double f0floor,
				    long nvc, long nvo, double mu,int mlt);
extern DVECTOR cspb_xgetiregion(DVECTOR gent, double t0, double mpv,
				double mu);
extern DVECTOR cspb_xgetwavelet(DVECTOR t, double t0, double mu, int mlt);
extern DMATRIX zwvlt2ifq(DMATRIX pm, double fs);
extern DMATRICES zifq2gpm2(DMATRIX pif, double f0floor, long nvo);
extern double *znrmlcf2(double f);
extern DVECTOR zGcBs(DVECTOR x, double k);
extern DMATRIX zsmoothmapB(DMATRIX map, double fs, double f0floor, long nvo,
			   double mu, double mlim, double pex);
extern DVECTORS zfixpfreq3(DVECTOR fxx, DVECTOR pif2, DVECTOR mmp,
			   DVECTOR dfv, DVECTOR pm, XBOOL allparam_flag);
extern DVECTORS plotcpower(DVECTOR x, double fs, double shiftm);
extern DVECTOR fir1bp(long len, double startf, double endf);
extern DVECTORS f0track5(DMATRICES f0infomat, DVECTORS pwinfo, double shiftm,
			 XBOOL allparam_flag);
extern DVECTORS refineF02(DVECTOR x, double fs, DVECTOR f0raw, long fftl,
			  double eta, long nhmx, double shiftm, long nl,
			  long nu, XBOOL allparam_flag);
extern DVECTORS refineF06(DVECTOR x, double fs, DVECTOR f0raw, long fftl,
			  double eta, long nhmx, double shiftm, long nl,
			  long nu, XBOOL allparam_flag);
extern DVECTOR interp1q(DVECTOR x, DVECTOR y, DVECTOR xi);
extern double *znrmlcf(double f);
extern DVECTOR getf0var(DVECTOR f0raw, DVECTOR irms);
extern void pruningf0(DMATRIX f0v, DMATRIX vrv, DVECTOR f0raw,
		      double maxf0, double minf0);
extern void plotcandf0file(DMATRIX f0v, DMATRIX vrv, DVECTOR f0raw,
			   char *cf0file, double f0ceil, double f0floor,
			   long f0shiftm);
extern void plotcandf0file_prun(DMATRIX f0v, DMATRIX vrv, DVECTOR f0raw,
				char *cf0file, double f0ceil, double f0floor,
				double maxf0, double minf0, long f0shiftl);
extern void plotcandf0file2(DMATRIX f0v, DMATRIX vrv, DVECTOR f0raw,
			    char *cf0file, double f0ceil, double f0floor,
			    long f0shiftm);
extern void plotcandf0file2_prun(DMATRIX f0v, DMATRIX vrv, DVECTOR f0raw,
				 char *cf0file, double f0ceil, double f0floor,
				 double maxf0, double minf0, long f0shiftl);
extern void writecandf0file(DVECTOR orgf0, DVECTOR extf0, char *cf0file,
			    long f0shiftl);
extern void quicksort(double *array, long lower, long upper, long *idx);



#endif /* __TEMPO_SUB_H */
