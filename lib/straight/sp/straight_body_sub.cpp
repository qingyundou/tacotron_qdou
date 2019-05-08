/*
 *	straight_body_sub.c : straight analysis subroutine
 *
 *		coded by K.Toyama 	1996/12/24
 *		modified by H.Banno	1996/12/28
 *		version 2.0 by H.Banno	1997/1/15
 *		version 3.0 by H.Banno	1997/3/12
 *
 *		straight_body_c03 V30k18 (matlab)
 *			by T. Toda	2001/2/10
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "defs.h"
#include "basic.h"
#include "matrix.h"
#include "voperate.h"
#include "complex.h"
#include "fft.h"
#include "filter.h"
#include "fileio.h"
#include "memory.h"

#include "straight_sub.h"
#include "tempo_sub.h"
#include "straight_body_sub.h"

/*
 *	extract fundamental frequency
 */
double sb_extractf0(		   	/* (r): f0 */
		    DVECTOR f0l,    	/* (i): f0 information */
		    long idf0, 	   	/* (i): index of f0 */
		    double fguard, 	/* (i): forward guard region */
		    double bguard, 	/* (i): backward guard region */
		    double f0shiftm)	/* (i): f0 shift length [ms] */
{
    double f0 = 0.0;			/* fundamental frequency */

    if (f0l->data[idf0] > 0.0) {	/* voiced frame */
	f0 = f0l->data[idf0];
    } 
    else {				/* unvoiced frame */
	long k;
	long len;			/* search length */
	long st, ed;			/* search region */
	long numv;			/* number of voiced frame */
	long numu;			/* number of unvoiced frame */
	double uratio;			/* ratio of unvoiced frame */
	double vf0;			/* f0 average of voiced frame */
	double sum;			/* sumation */

	/* set forward search region */
	st = idf0;
	ed = MIN(idf0 + (long)(fguard / f0shiftm), f0l->length - 1);

	len = ed - st + 1;
	numv = 0;
	numu = 0;
	sum = 0.0;
	for (k = st; k <= ed; k++) {
	    sum += f0l->data[k];
	    if (f0l->data[k] == 0.0) {
		numu++;
	    } else {
		numv++;
	    }
	}

	if (sum > 0.0) {
	    uratio = (double)numu / (double)len;
	    vf0 = (numv <= 0 ? 0.0 : sum / (double)numv);
	    f0 = (UNVOICED_F0 - vf0) * SQUARE(uratio) + vf0;
	} else {
	    /* set backward search region */
	    st = MAX(idf0 - (long)(bguard / f0shiftm), 0);
	    ed = idf0;

	    len = ed - st + 1;
	    numv = 0;
	    numu = 0;
	    sum = 0.0;
	    for (k = st; k <= ed; k++) {
		sum += f0l->data[k];
		if (f0l->data[k] == 0.0) {
		    numu++;
		} else {
		    numv++;
		}
	    }

	    if (sum > 0.0) {
		uratio = (double)numu / (double)len;
		vf0 = (numv <= 0 ? 0.0 : sum / (double)numv);
		f0 = (UNVOICED_F0 - vf0) * SQUARE(uratio) + vf0;
	    } else {
		f0 = UNVOICED_F0;
	    }
	}
    }
    
    return f0;
}

DVECTOR sb_xwintable(
		     long framel,
		     double *refw)
{
    long k;
    double value;
    double sum;
    DVECTOR wt = NODATA;

    /* make window table for cut sample */
    wt = xdvalloc(framel);
    for (k = 0, sum = 0.0; k < framel; k++){
	/* value = 4*[-1:1] */
	value = 8.0 * ((double)(k + 1) - (double)framel / 2.0) / (double)framel;
	wt->data[k] = exp(-SQUARE(value) / 2.0);
	sum += wt->data[k];
    }
    *refw = sum;

    return wt;
}

DVECTOR sb_xgetpswin(			/* (r): pitch synchronous window */
		     DVECTOR wt, 	/* (i): window table */
		     double t0,		/* (i): pitch period */
		     double refw)
{
    DVECTOR ww = NODATA;		/* pitch synchronous window */

    /* adjust window shape */
    ww = xdvclone(wt);
    dvscoper(ww, "^", SQUARE(refw / t0));
    dvscoper(ww, "/", dvsum(ww));

    return ww;
}

DVECTOR sb_xcutsig(			/* (r): cut signal */
		   DVECTOR sig, 	/* (i): input signal */
		   long offset, 	/* (i): data offset */
		   long length)		/* (i): data length */
{
    DVECTOR cx = NODATA;

    /* cut signal */
    cx = xdvcut(sig, offset, length);
    dvscoper(cx, "-", dvmean(cx));

    return cx;
}

DVECTOR sb_xgetfftpow(			/* (r): fft power spectrum */
		      DVECTOR cx, 	/* (i): 1 frame sample */
		      DVECTOR wxe, 	/* (i): window for cutting sample */
		      long fftl,	/* (i): fft point */	
		      double pc) 	/* (i): power constant */
{
    DVECTOR cx2 = NODATA;
    DVECTOR pw = NODATA;			/* fft power spectrum */

    /* get windowed data of tx */
    cx2 = xdvclone(cx);
    dvoper(cx2, "*", wxe);

    /* get fft power spectrum */
    pw = xdvfftabs(cx2, fftl);

    /* pw = pw ^ pc */
    dvscoper(pw, "^", pc);

    /* memory free */
    xdvfree(cx2);

    return pw;
}

DVECTOR sb_xgetfftangle(		/* (r): fft phase angle */
			DVECTOR cx, 	/* (i): 1 frame sample */
			DVECTOR wxe, 	/* (i): window for cutting sample */
			long fftl)	/* (i): fft point */	
{
    DVECTOR cx2 = NODATA;
    DVECTOR phs = NODATA;			/* fft phase angle */

    /* get windowed data of tx */
    cx2 = xdvclone(cx);
    dvoper(cx2, "*", wxe);

    /* get fft phase angle */
    phs = xdvfftangle(cx2, fftl);

    /* memory free */
    xdvfree(cx2);

    return phs;
}

DVECTOR sb_xgetfftgrpdly(		/* (r): fft group delay */
			 DVECTOR cx, 	/* (i): 1 frame sample */
			 DVECTOR wxe, 	/* (i): window for cutting sample */
			 long fftl)	/* (i): fft point */	
{
    DVECTOR cx2 = NODATA;
    DVECTOR phs = NODATA;			/* fft phase angle */

    /* get windowed data of tx */
    cx2 = xdvclone(cx);
    dvoper(cx2, "*", wxe);

    /* get fft phase angle */
    phs = xdvfftgrpdly(cx2, fftl);

    /* memory free */
    xdvfree(cx2);

    return phs;
}

DVECTOR sb_xsmoothfreq(			/* (r): smoothed data */
		       DVECTOR pw, 	/* (i): fft power spectrum */
		       double t0,	/* (i): fundamental frequency */
		       long fftl)	/* (i): fft point */
{
    long offset;
    DVECTOR ww = NODATA;
    DVECTOR ww2 = NODATA;
    DVECTOR spw = NODATA;

    /* memory allocation for convolution */
    ww2 = xdvzeros(fftl);

    /* calculate spectral smoothing window */
    ww = sb_xgetsmoothwin((double)fftl / t0);

    /* copy data */
    offset = (fftl - ww->length + 1) / 2;	/* because ww->length is odd */
    dvpaste(ww2, ww, offset, ww->length, 0);
    dvfftshift(ww2);

    /* fft convolution */
    spw = xdvfftconv(ww2, pw, fftl);

    /* memory free */
    xdvfree(ww);
    xdvfree(ww2);

    return spw;
}

DVECTOR sb_xgetsmoothwin(		/* (r): smoothing window */
			 double t0)	/* (i): f0 in point */ 
{
    long k;
    long t0l;
    DVECTOR ww = NODATA;

    /* initialize */
    t0l = (long)floor(t0);
    ww = xdvalloc(2 * t0l + 1);

    ww->data[t0l] = 1.0;
    for (k = 0; k < t0l; k++){
	ww->data[k] = 1.0 - (double)(t0l - k) / t0;
	ww->data[ww->length - 1 - k] = ww->data[k];
    }

    /* normalize ww */
    dvscoper(ww, "/", dvsum(ww));

    return ww;
}

/*
 *	function for Version 2.0
 */
DVECTOR sb_xgetsinglewin(
			 double t0, 	/* (i): pitch period in point */
			 long framel,
			 double *cf)
{
    long k;
    double idx;
    double sum;
    DVECTOR wxs = NODATA;

    /* initialize */
    wxs = xdvalloc(framel);

    for (k = 0, sum = 0.0; k < framel; k++){
	idx = (double)(k + 1 - framel / 2) / t0;
	wxs->data[k] = exp(-PI * SQUARE(idx));
	sum += ABS(wxs->data[k]);
    }
    dvscoper(wxs, "/", sum);
    if (cf != NULL) {
	*cf = sum;
    }

    return wxs;
}

DVECTOR sb_xgetdoublewin(
			 double t0, 	/* (i): pitch period in point */
			 long framel, 	/* (i): frame length */
			 double cf)
{
    long k;
    double a, b;
    double idx;
    double sum;
    DVECTOR wxd = NODATA;

    /* initialize */
    wxd = xdvalloc(framel);

    for (k = 0, sum = 0.0; k < framel; k++){
	idx = (double)(k + 1 - framel / 2) / t0;
	a = idx - 0.5;
	b = idx + 0.5;
	wxd->data[k] = exp(-PI * SQUARE(a)) - exp(-PI * SQUARE(b));
	sum += ABS(wxd->data[k]);
    }
    if (cf > 0.0) {
	dvscoper(wxd, "/", cf * 2.0);
    } else {
	dvscoper(wxd, "/", sum);
    }

    return wxd;
}

DVECTOR sb_xgetdbfftpow(DVECTOR pws, DVECTOR pwd, double pc)
{
    long k;
    double value;
    DVECTOR pw = NODATA;

    pw = xdvalloc(MIN(pws->length, pwd->length));

    for (k = 0; k < pw->length; k++) {
	if (pc == 2.0) {
	    pw->data[k] = pws->data[k] + pwd->data[k];
	} else {
	    value = sqrt(pws->data[k] + pwd->data[k]);
	    pw->data[k] = pow(value, pc);
	}
    }

    return pw;
}

DVECTOR sb_xleveleqspec(		/* (r): level equalize spectrum */
			DVECTOR pw, 	/* (i): fft power spectrum */
			double t0, 	/* (i): pitch period in point */
			long fftl)	/* (i): fft point */
{
    long k;
    long fftl2;
    double idx;
    double sum;
    DVECTOR ww2 = NODATA;
    DVECTOR spw2 = NODATA;

    /* initailize */
    fftl2 = fftl / 2;
    ww2 = xdvalloc(fftl);

    /* make spectral smoothing window */
    for (k = 0, sum = 0.0; k < fftl; k++) {
	idx = (double)(k - fftl2) / (double)fftl * t0;
	ww2->data[k] = MAX(1.0 - FABS(idx / 3.0), 0);
	sum += ww2->data[k];
    }
    dvscoper(ww2, "/", sum);
    dvfftshift(ww2);

    /* convolution */
    spw2 = xdvfftconv(ww2, pw, fftl);

    /* memory free */
    xdvfree(ww2);

    return spw2;
}

/*
 *	smoothing window using spline theory
 */
DVECTOR sb_xgetsmoothwin_s(		/* (r): smoothing window */
			   double t0,	/* (i): f0 in point */ 
			   long fftl)	/* (i): fft point */
{
    long k;
    long fftl2;
    double a, b, c;
    double idx;
    double sum;
    DVECTOR ww = NODATA;

    /* initialize */
    fftl2 = fftl / 2;
    ww = xdvalloc(fftl);

    /* make spectral smoothing window */
    for (k = 0, sum = 0.0; k < fftl; k++) {
	idx = (double)(k - fftl2) / (double)fftl * t0;
	a = MAX(1.0 - FABS(idx), 0);
	b = MAX(1.0 - FABS(idx - 1.0), 0);
	c = MAX(1.0 - FABS(idx + 1.0), 0);

	ww->data[k] = 1.4214 * a - 0.2107 * (b + c);
	sum += ww->data[k];
    }
    dvscoper(ww, "/", sum);

    return ww;
}

/*
 *	frequency smoothing using spline theory
 */
DVECTOR sb_xsmoothfreq_s(		/* (r): smoothed data */
			 DVECTOR pw, 	/* (i): fft power spectrum */
			 double t0,	/* (i): fundamental frequency */
			 long fftl)	/* (i): fft point */
{
    long k;
    double a, b;
    DVECTOR ww = NODATA;
    DVECTOR pw2 = NODATA;
    DVECTOR spw = NODATA;
    DVECTOR spw2 = NODATA;

    /* spectrum for local level equalization */
    spw2 = sb_xleveleqspec(pw, t0, fftl);

    /* calculate spectral smoothing window */
    ww = sb_xgetsmoothwin_s(t0, fftl);
    dvfftshift(ww);

    /* fft convolution */
    pw2 = xdvoper(pw, "/", spw2);
    spw = xdvfftconv(ww, pw2, fftl);

    /* half-wave rectification */
    for (k = 0; k < spw->length; k++) {
	a = spw->data[k] * 4.0;
	b = 0.25 * (FABS(a) + exp(-FABS(a)) + a) / 2.0;
	spw->data[k] = b * spw2->data[k];
    }

    /* memory free */
    xdvfree(ww);
    xdvfree(pw2);
    xdvfree(spw2);

    return spw;
}

DVECTOR sb_xtimeexpand(DVECTOR cumfreq, LVECTOR tx, long ii, long nframe)
{
    long k;
    long pos;
    DVECTOR txx = NODATA;

    txx = xdvalloc(tx->length);
    for (k = 0; k < tx->length; k++) {
	pos = ii + tx->data[k];
	pos = MAX(0, pos);
	pos = MIN(nframe - 1, pos);
	txx->data[k] = cumfreq->data[pos] - cumfreq->data[ii];
    }

    return txx;
}

LVECTOR sb_xfindregion(DVECTOR txx, double value, int eqflag)
{
    LVECTOR idx = NODATA;
    DVECTOR txxa = NODATA;

    txxa = xdvabs(txx);
    if (eqflag == 1) {
	idx = xdvscfind(txxa, "<=", value);
    } else {
	idx = xdvscfind(txxa, "<", value);
    }
    xdvfree(txxa);

    return idx;
}

/*
 *	function for Version 3.0
 */
DVECTOR sb_xgetsinglewin2(
			  double t0, 	/* (i): pitch period in point */
			  long framel,
			  double coef)
{
    long k;
    double idx;
    double sum;
    DVECTOR wxs = NODATA;

    /* initialize */
    wxs = xdvalloc(framel);
    if (coef < 1.0) {
	coef = 1.0;
    }

    for (k = 0, sum = 0.0; k < framel; k++){
	idx = (double)(k + 1 - framel / 2) / t0 / coef;
	wxs->data[k] = exp(-PI * SQUARE(idx));
	sum += SQUARE(wxs->data[k]);
    }

    /* normalize window */
    sum = sqrt(sum);
    dvscoper(wxs, "/", sum);

    return wxs;
}

DVECTOR sb_xgetdoublewin2(		/* (r): double window */
			  DVECTOR wxs, 	/* (i): single window */
			  double t0, 	/* (i): pitch period in point */
			  long framel, 	/* (i): frame length */
			  double coef) 	/* (i): coefficient */
{
    long k;
    double idx;
    DVECTOR wxd = NODATA;

    /* initialize */
    wxd = xdvalloc(framel);

    for (k = 0; k < framel; k++){
	idx = (double)(k + 1 - framel / 2) / t0;
	wxd->data[k] = coef * wxs->data[k] * sin(PI * idx);
    }

    return wxd;
}

/*
 *	smoothing window using spline theory
 */
DVECTOR sb_xgetsmoothwin_s2(		/* (r): smoothing window */
			    double t0,	/* (i): f0 in point */ 
			    long fftl)	/* (i): fft point */
{
    long k;
    long fftl2;
    double a, b, c, d, e;
    double idx;
    double sum;
    DVECTOR ww = NODATA;

    /* initialize */
    fftl2 = fftl / 2;
    ww = xdvalloc(fftl);

    /* make spectral smoothing window */
    for (k = 0, sum = 0.0; k < fftl; k++) {
	idx = (double)(k - fftl2) / (double)fftl * t0;
	a = MAX(1.0 - FABS(idx), 0);
	b = MAX(1.0 - FABS(idx - 1.0), 0);
	c = MAX(1.0 - FABS(idx + 1.0), 0);
	d = MAX(1.0 - FABS(idx - 2.0), 0);
	e = MAX(1.0 - FABS(idx + 2.0), 0);

	ww->data[k] = 1.6495 * a - 0.4031 * (b + c) + 0.0985 * (d + e);
	sum += ww->data[k];
    }
    dvscoper(ww, "/", sum);

    return ww;
}

void sb_halfrectspec(DVECTOR spw)	/* (i/o): fft power spectrum */
{
    long k;
    double a;

    /* half-wave rectification */
    for (k = 0; k < spw->length; k++) {
	a = spw->data[k] * 4.0;
	spw->data[k] = 0.25 * (log(2.0 * cosh(a / 1.4)) * 1.4 + a) / 2.0;
    }

    return;
}

/*
 *	frequency smoothing using spline theory
 */
DVECTOR sb_xsmoothfreq_s2(		/* (r): smoothed data */
			  DVECTOR pw, 	/* (i): fft power spectrum */
			  double t0,	/* (i): fundamental frequency */
			  long fftl)	/* (i): fft point */
{
    DVECTOR ww = NODATA;
    DVECTOR pw2 = NODATA;
    DVECTOR spw = NODATA;
    DVECTOR spw2 = NODATA;

    /* spectrum for local level equalization */
    spw2 = sb_xleveleqspec(pw, t0, fftl);

    /* calculate spectral smoothing window */
    ww = sb_xgetsmoothwin_s2(t0, fftl);
    dvfftshift(ww);

    /* fft convolution */
    pw2 = xdvoper(pw, "/", spw2);
    spw = xdvfftconv(ww, pw2, fftl);

    /* half-wave rectification */
    sb_halfrectspec(spw);
    dvoper(spw, "*", spw2);

    /* memory free */
    xdvfree(ww);
    xdvfree(pw2);
    xdvfree(spw2);

    return spw;
}

DVECTOR sb_cutwindhigh = NODATA;
DVECTOR sb_cutwindlow = NODATA;

void sb_blendspec(DVECTOR spw, 
		  DVECTOR pw, 
		  double lambh,
		  double lambl,
		  double fs,
		  long fftl)
{
    static long last_fftl = 0;
    long k;
    long hfftl;
    double pwh, pwl;
    double lambh2, lambl2;

    hfftl = fftl / 2 + 1;

    if (last_fftl != fftl) {
	if (sb_cutwindhigh != NODATA) {
	    xdvfree(sb_cutwindhigh);
	}
	if (sb_cutwindlow != NODATA) {
	    xdvfree(sb_cutwindlow);
	}
	sb_cutwindhigh = sb_xlowcutwin(fs, 100.0, 600.0, fftl);
	sb_cutwindlow = xdvscoper(sb_cutwindhigh, "!-", 1.0);
    }

    lambh2 = 1.0 - lambh;
    lambl2 = 1.0 - lambl;

    for (k = 0; k < hfftl; k++) {
	pwh = lambh * spw->data[k] + lambh2 * pw->data[k];
	pwl = lambl * spw->data[k] + lambl2 * pw->data[k];
	spw->data[k] = sb_cutwindhigh->data[k] * pwh + sb_cutwindlow->data[k] * pwl;
    }
    dvfftturn(spw);

    return;
}

DVECTOR sb_xlowcutwin(double fs, double bw, double cornf, long fftl) 
{
    long k; 
    DVECTOR cww = NODATA;

    cww = xdvalloc(fftl / 2 + 1);
    for (k = 0; k < cww->length; k++) {
	cww->data[k] = 1.0 / 
	    (1.0 + exp(-((double)(k + 1) * fs / (double)fftl - cornf) / bw));
    }

    return cww;
}

/*
 *	memory free 
 */
void sb_xfree_sub(void)
{
    if (sb_cutwindhigh != NODATA) {
	xdvfree(sb_cutwindhigh);
    }
    if (sb_cutwindlow != NODATA) {
	xdvfree(sb_cutwindlow);
    }

    return;
}

/* Butterworth digital and analog high-pass filter design */
DVECTORS butter(long n,		/* order */
		double cutoff)	/* cut-off frequency */

{
    long k;
    double gain = 1.0;
    DVECTOR pv = NODATA;
    DVECTORS filter = NODATA;

    /* 0 <= cufoff <= 1 */
    cutoff = MIN(cutoff, 1.0);
    cutoff = MAX(cutoff, 0.0);

    /* 2 <= n <= 16 */
    if (n < 2 || n > 16) {
	printmsg(stderr, "2 <= Order <= 16\n");
	return NODATA;
    }

    /* calculate gain and pole of chebyshev filter */
    pv = buttap(n);

    /* highpass filter */
    for (k = 0; k + 1< n; k += 2) {
	gain *= pow(pv->data[k], 2.0) + pow(pv->imag[k], 2.0);
    }
    if (k < n) gain *= -1.0 * pv->data[k];
    gain /= 1.0;
    dvscoper(pv, "^", -1.0);
    dvscoper(pv, "*", cutoff);
   
    /* convert analog into digital through bilinear transformation */
    filter = a2dbil(pv, gain, cutoff, XTRUE);	/* highpass filter */

    /* memory free */
    xdvfree(pv);

    return filter;
}

/* Butterworth analog lowpass filter prototype (gain = 1.0) */
DVECTOR buttap(long n)	/* order */
{
    long l, l2;
    double w;
    DVECTOR pv = NODATA;
    
    /* calculate pole of butterworth filter */
    pv = xdvrialloc(n);
    for (l = 0; l < n / 2; l++) {
	w = ((double)n + 1.0 + 2.0 * (double)l) / (2.0 * (double)n) * PI;
	l2 = l * 2;
	pv->data[l2] = cos(w);
	pv->imag[l2] = sin(w);
	pv->data[l2 + 1] = pv->data[l2];
	pv->imag[l2 + 1] = -1.0 * pv->imag[l2];
    }
    if ((l2 = l * 2) < n) {
	w = ((double)n + 1.0 + 2.0 * (double)l) / (2.0 * (double)n) * PI;
	pv->data[l2] = cos(w);
	pv->imag[l2] = 0.0;
    }

    return pv;
}

/* Calculate the optimum smoothing function */
DVECTOR optimumsmoothing(double eta,	/* temporal stretch factor */
			 double pc)	/* power exponent for nonlinearity */
{
    long k, bbc, nn, ii, idc;
    double fx, max;
    LVECTOR ss = NODATA;	DVECTOR cb = NODATA;
    DVECTOR gw = NODATA;	DVECTOR cmwb = NODATA;
    DVECTOR cmw = NODATA;	DVECTOR cmws = NODATA;
    DVECTOR ovc = NODATA;
    DMATRIX bvm = NODATA;	DMATRIX hh = NODATA;
    DMATRIX hht = NODATA;	DMATRIX h = NODATA;
    DMATRIX invh = NODATA;	DMATRIX tmp = NODATA;
    DMATRIX ov = NODATA;

    /* make window */
    cb = xdvalloc((long)((double)(8 +8) / 0.05) + 1);
    gw = xdvalloc(cb->length);
    for (k = 0, fx = -8.0; k < cb->length; k++, fx += 0.05) {
	/* triangular window */
	cb->data[k] = MAX(0.0, 1.0 - fabs(fx));
	/* gaussian window */
	gw->data[k] = pow(exp(-PI * SQUARE(fx * eta)), pc);
    }
    /* convolution */
    cmwb = xdvconv(cb, gw);
    bbc = (cb->length - 1) / 2;
    max = dvmax(cmwb, NULL);
    cmw = xdvalloc(cb->length);
    ss = xlvalloc(cb->length);
    for (k = 0, fx = -8.0, nn = 0; k < cb->length; k++, fx += 0.05) {
	cmw->data[k] = cmwb->data[k + bbc] / max;
	if (fabs(fx - round(fx)) < 0.025) {
	    ss->data[nn] = k;
	    nn++;
	}
    }
    cmws = xdvalloc(nn);
    for (k = 0; k < nn; k++) cmws->data[k] = cmw->data[ss->data[k]];
    /* memory free */
    xlvfree(ss);	xdvfree(cb);
    xdvfree(gw);	xdvfree(cmwb);	xdvfree(cmw);

    /* memory allocation */
    hh = xdmzeros(2 * nn, nn);
    for (ii = 0; ii < nn; ii++) dmpastecol(hh, ii, cmws, ii, cmws->length, 0);
    /* memory free */
    xdvfree(cmws);
    
    /* memory allocation */
    bvm = xdmzeros(2 * nn, 1);
    /* This is the original unit impulse. */
    bvm->data[nn][0] = 1.0;
    hht = ss_xtrans_mat(hh);
    tmp = ss_xmulti_mat(hht, bvm);
    h = ss_xmulti_mat(hht, hh);
    invh = ss_xinvmat_svd(h, 1.0e-12);
    /* This is the optimum coefficient vector. */
    ov = ss_xmulti_mat(invh, tmp);
    /* memory free */
    xdmfree(bvm);	xdmfree(hh);	xdmfree(hht);
    xdmfree(h);		xdmfree(invh);	xdmfree(tmp);
    
    ovc = xdvalloc(4);
    idc = (nn - 1) / 2 + 1;
    for (ii = 0; ii < ovc->length; ii++) ovc->data[ii] = ov->data[idc + ii][0];
    /* memory free */
    xdmfree(ov);

    return ovc;
}

/*
 *	frequency smoothing using spline theory
 * 	straight_body_c03	V30kr18
 */
DVECTOR sb_xsmoothfreq_c03(		/* (r): smoothed data */
			   DVECTOR pw, 	/* (i): fft power spectrum */
			   double t0,	/* (i): fundamental frequency */
			   long fftl,	/* (i): fft point */
			   DVECTOR ovc)	/* coefficient for smooth win */
{
    long k, fftl2;
    double idx, idx2;
    DVECTOR ww2t = NODATA;
    DVECTOR wwt = NODATA;
    DVECTOR pw2 = NODATA;
    DVECTOR spw = NODATA;
    DVECTOR spw2 = NODATA;
    DVECTOR tmp = NODATA;

    /* initailize */
    fftl2 = fftl / 2;
    wwt = xdvalloc(fftl);
    ww2t = xdvalloc(fftl);

    /* make spectral smoothing window (Time domain) */
    wwt->data[0] = ovc->data[0] + ovc->data[1] * 2.0 + ovc->data[2] * 2.0;
    ww2t->data[0] = 1.0;
    for (k = 1; k < fftl2 + 1; k++) {
	idx = (double)k / t0 * PI;	idx2 = idx * 3.0;
	wwt->data[k] = SQUARE(sin(idx) / idx) * (ovc->data[0] + ovc->data[1] * 2.0 * cos(idx * 2.0) + ovc->data[2] * 2.0 * cos(idx * 4.0));
	ww2t->data[k] = SQUARE(sin(idx2) / idx2);
    }
    for (; k < fftl; k++) {
	idx = (double)(-fftl + k) / t0 * PI;	idx2 = idx * 3.0;
	wwt->data[k] = SQUARE(sin(idx) / idx) * (ovc->data[0] + ovc->data[1] * 2.0 * cos(idx * 2.0) + ovc->data[2] * 2.0 * cos(idx * 4.0));
	ww2t->data[k] = SQUARE(sin(idx2) / idx2);
    }

    /* spectrum for local level equalization (convolution) */
    tmp = xdvfft(pw, fftl);
    dvoper(tmp, "*", ww2t);
    spw2 = xdvifft(tmp, fftl);
    dvreal(spw2);	xdvfree(tmp);

    /* smoothing (convolution) */
    pw2 = xdvoper(pw, "/", spw2);
    tmp = xdvfft(pw2, fftl);
    dvoper(tmp, "*", wwt);
    spw = xdvifft(tmp, fftl);
    dvreal(spw);	xdvfree(tmp);
    dvscoper(spw, "/", wwt->data[0]);

    /* half-wave rectification */
    sb_halfrectspec(spw);
    dvoper(spw, "*", spw2);

    /* memory free */
    xdvfree(wwt);
    xdvfree(ww2t);
    xdvfree(pw2);
    xdvfree(spw2);

    return spw;
}

void sb_xsmoothtime_c03(DMATRIX n2sgram,
			DVECTOR f0l,
			double lowestf0,
			double shiftm)
{
    long njj, jj, ii, id;
    long nframe;
    long tunitw, len;
    double sum;
    LVECTOR idx = NODATA;
    DVECTOR cumfreq = NODATA;
    DVECTOR txx = NODATA;
    DVECTOR wt = NODATA;
    DVECTOR spw = NODATA;
    DVECTOR pw = NODATA;
    DMATRIX nssgram = NODATA;
    
    njj = n2sgram->row;
    nframe = MIN(f0l->length, njj);
    cumfreq = xdvcumsum(f0l);
    dvscoper(cumfreq, "/", 1000.0 / shiftm);
    tunitw = (long)ceil(1.1 * (1000.0 / shiftm) / lowestf0);

    /* memory allocation */
    idx = xlvalloc(2 * tunitw + 1);
    txx = xdvalloc(2 * tunitw + 1);
    nssgram = ss_xdmclone(n2sgram);

    /* smoothing */
    for (jj = 0; jj < nframe; jj++) {
	for (ii = 0, len = 0; ii < 2 * tunitw + 1; ii++) {
	    id = MIN(njj - 1, MAX(0, jj + ii - tunitw));
	    txx->data[ii] = cumfreq->data[id] - cumfreq->data[jj];
	    if (ii - tunitw + jj >= 0 && ii - tunitw + jj < njj) {
		if (fabs(txx->data[ii]) < 1.1) {
		    idx->data[len] = ii;
		    len++;
		}
	    }
	}

	/* memory allocation */
	wt = xdvalloc(len);
	spw = xdvzeros(nssgram->col);

	/* convolution */
	for (ii = 0, sum = 0.0; ii < len; ii++) {
	    /* smooting window */
	    wt->data[ii] = MAX(0.0, 1.0 - fabs(txx->data[idx->data[ii]]));
	    id = MIN(njj - 1, MAX(0, jj - tunitw + idx->data[ii] + 1));
	    wt->data[ii] *= f0l->data[id];
	    wt->data[ii] *= (fabs(f0l->data[jj] - f0l->data[jj - tunitw + idx->data[ii]]) / f0l->data[jj] < 0.25);
	    sum += wt->data[ii];

	    id = MIN(njj - 1, MAX(0, jj - tunitw + idx->data[ii]));
	    pw = xdmextractrow(nssgram, id);
	    dvscoper(pw, "*", wt->data[ii]);
	    dvoper(spw, "+", pw);
	    /* memory free */
	    xdvfree(pw);
	}
	dvscoper(spw, "/", sum);
	dmcopyrow(n2sgram, jj, spw);

	/* memory free */
	xdvfree(wt);
	xdvfree(spw);
    }

    /* memory free */
    xlvfree(idx);
    xdvfree(txx);
    xdmfree(nssgram);
}

/* Dirty hack for controling time constant in unvoiced part analysis */
DVECTOR sb_xtconstuv_c03(DMATRIX n2sgram,
			 DVECTOR xh2,	/* input signal (more than 300 Hz) */
			 DVECTOR xhh,	/* input signal (more than 3000 Hz) */
			 DVECTOR sumvec,
			 double ttlv,
			 double ttlv2,
			 double fs,
			 double shiftm,
			 long ncw)
{
    long ipl, k;
    double shiftl, mmaa, ipwm, sum;
    DVECTOR h3 = NODATA;
    DVECTOR pwc = NODATA;
    DVECTOR pwch = NODATA;
    DVECTOR hann = NODATA;
    DVECTOR pwchz = NODATA;
    DVECTOR dpwchz = NODATA;
    DVECTOR apwtb = NODATA;
    DVECTOR apwt = NODATA;
    DVECTOR dpwtb = NODATA;
    DVECTOR dpwt = NODATA;
    DVECTOR rr = NODATA;
    DVECTOR lmbd = NODATA;

    shiftl = shiftm * fs / 1000.0;

    /* make window */
    h3 = xgethannexpwin(fs, ncw);

    /* power of input signal (more than 300 Hz) */
    pwc = xgetpowsig(xh2, h3, ttlv2, shiftl, ncw, n2sgram->row);

    /* power of input signal (more than 3000 Hz) */
    pwch = xgetpowsig(xhh, h3, ttlv, shiftl, ncw, n2sgram->row);

    /* impact detection window size */
    ipwm = 7.0;
    ipl = (long)round(ipwm / shiftm);
    /* Hanning window */
    hann = xdvalloc(ipl * 2 + 1);
    for (k = 0, sum = 0.0; k < hann->length; k++) {
	hann->data[k] = 0.5 - 0.5 * cos(2.0 * (double)(k + 1)* PI / (double)(hann->length + 1));
	sum += hann->data[k];
    }
    dvscoper(hann, "/", sum);

    /* smoothing of power (more than 3000 Hz) */
    pwchz = xdvalloc(pwch->length + hann->length * 2);
    for (k = 0; k < pwch->length; k++) pwchz->data[k] = pwch->data[k];
    for (; k < pwchz->length; k++) pwchz->data[k] = 0.0;
    apwtb = xdvfftfiltm(hann, pwchz, hann->length * 2);
    apwt = xdvcut(apwtb, ipl, pwch->length);

    /* smoothing of differential power (more than 300 Hz) */
    dpwchz = xdvalloc(pwch->length - 1 + hann->length * 2);
    for (k = 0; k < pwch->length - 1; k++)
	dpwchz->data[k] = SQUARE(pwch->data[k + 1] - pwch->data[k]);
    for (; k < dpwchz->length; k++) dpwchz->data[k] = 0.0;
    dpwtb = xdvfftfiltm(hann, dpwchz, hann->length * 2);
    dpwt = xdvcut(dpwtb, ipl, pwch->length);

    /* calculate time constant controller */
    mmaa = dvmax(apwt, NULL);
    rr = xdvalloc(pwch->length);
    lmbd = xdvalloc(pwch->length);
    for (k = 0; k < pwch->length; k++) {
	/* floating error */
        if (dpwt->data[k] <= 0.0) dpwt->data[k] = 0.0;
	if (apwt->data[k] <= 0.0) {
	    rr->data[k] = sqrt(dpwt->data[k]) / mmaa;
	} else {
	    rr->data[k] = sqrt(dpwt->data[k]) / apwt->data[k];
	}
	lmbd->data[k] = 1.0 / (1.0 + exp(-(sqrt(rr->data[k]) - 0.75) * 20.0));
	/* time constant controller */
	pwc->data[k] = pwc->data[k] * lmbd->data[k] + (1.0 - lmbd->data[k]) * sumvec->data[k];
    }

    /* memory free */
    xdvfree(h3);	xdvfree(hann);
    xdvfree(pwchz);	xdvfree(apwtb);	xdvfree(dpwchz);
    xdvfree(dpwtb);	xdvfree(pwch);	xdvfree(apwt);
    xdvfree(dpwt);	xdvfree(rr);	xdvfree(lmbd);

    return pwc;
}

DVECTOR xgethannexpwin(double fs,
		       long ncw)
{
    long k;
    DVECTOR hann = NODATA;
    DVECTOR expv = NODATA;
    DVECTOR h3 = NODATA;

    /* Hanning window */
    hann = xdvalloc((long)round(fs / 1000.0));
    for (k = 0; k < hann->length; k++) 
	hann->data[k] = 0.5 - 0.5 * cos(2.0 * (double)(k + 1) * PI / (double)(hann->length + 1));

    /* exponential window */
    expv = xdvalloc(ncw * 2 + 1);
    for (k = 0; k <= ncw * 2; k++)
	expv->data[k] = exp(-1400.0 / fs * (double)k);

    /* convolution */
    h3 = xdvconv(hann, expv);

    /* memory free */
    xdvfree(hann);
    xdvfree(expv);

    return h3;
}

DVECTOR xgetpowsig(DVECTOR x,
		   DVECTOR w,
		   double ttlv,
		   double shiftl,
		   long ncw,
		   long len)
{
    long k;
    double sum;
    DVECTOR x2 = NODATA;
    DVECTOR pwc = NODATA;
    DVECTOR pwcb = NODATA;

    /* memory allocation */
    x2 = xdvalloc(x->length + ncw * 10);

    /* caluculate power */
    for (k = 0; k < x->length; k++)
	x2->data[k] = SQUARE(fabs(x->data[k]));
    for (; k < x2->length; k++) x2->data[k] = 0.0;

    /* filtering */
    pwcb = xdvfftfiltm(w, x2, w->length * 2);
    pwc = xdvalloc(len);
    for (k = 0, sum = 0.0; k < len; k++) {
	pwc->data[k] = pwcb->data[(long)round((double)k * shiftl)];
	sum += pwc->data[k];
    }
    dvscoper(pwc, "*", ttlv / sum);

    /* memory free */
    xdvfree(x2);
    xdvfree(pwcb);

    return pwc;
}

/* Spectral compensation using Time Domain technique */
void specreshape(DMATRIX n2sgram,	/* Straight smoothed spectrogram */
		 DVECTOR f0l,		/* fundamental frequency (Hz) */
		 double fs,		/* sampling frequency (Hz) */
		 double eta,		/* temporal stretch factor */
		 double pc,		/* power exponent for nonlinearity */
		 double mag,		/* magnification factor */
		 XBOOL msg_flag)
{
    long fftl, hfftl, ii, k;
    double sumbb;
    DVECTOR pb2 = NODATA;
    DVECTOR ovc = NODATA;
    DVECTOR ffs = NODATA;
    DVECTOR ffsb = NODATA;
    DVECTOR ccs2 = NODATA;
    DVECTOR ngg = NODATA;
    DMATRIX hh = NODATA;
    DMATRIX ihh = NODATA;
    DMATRIX ovcm = NODATA;
    DMATRIX bb = NODATA;

    fftl = (n2sgram->col - 1) * 2;
    hfftl = n2sgram->col;

    /* memory allocation */
    pb2 = xdvalloc(fftl);
    hh = xdmalloc(4, 4);
    hh->data[0][0] = 1.0;	hh->data[0][1] = 1.0;
    hh->data[0][2] = 1.0;	hh->data[0][3] = 1.0;
    hh->data[1][0] = 0.0;	hh->data[1][1] = 1.0 / 2.0;
    hh->data[1][2] = 2.0 / 3.0;	hh->data[1][3] = 3.0 / 4.0;
    hh->data[2][0] = 0.0;	hh->data[2][1] = 0.0;
    hh->data[2][2] = 1.0 / 3.0;	hh->data[2][3] = 2.0 / 4.0;
    hh->data[3][0] = 0.0;	hh->data[3][1] = 0.0;
    hh->data[3][2] = 0.0;	hh->data[3][3] = 1.0 / 4.0;

    ovc = optimumsmoothing(eta, pc);

    ihh = ss_xinvmat_svd(hh, 1.0e-12);

    ovcm = ss_xvec2matcol(ovc);
    bb = ss_xmulti_mat(ihh, ovcm);

    sumbb = bb->data[0][0] + 4.0 * bb->data[1][0]
	+ 9.0 * bb->data[2][0] + 16.0 * bb->data[3][0];
    for (ii = 0; ii < fftl; ii++) {
	pb2->data[ii] = SQUARE((double)ii / fs);
	pb2->data[ii] *= PI / SQUARE(eta) + SQUARE(PI) / 3.0 * sumbb;
    }

    /* memory free */
    xdvfree(ovc);	xdmfree(ovcm);
    xdmfree(hh);	xdmfree(ihh);	xdmfree(bb);

    /* convolution */
    for (ii = 0; ii < n2sgram->row; ii++) {
	/* spectrum */
	ffsb = xdmextractrow(n2sgram, ii);
	ffs = xdvalloc(fftl);
	dvcopy(ffs, ffsb);
	dvfftturn(ffs);

	/* FFT convolution */
	ccs2 = xdvfft(ffs, ffs->length);
	dvreal(ccs2);
	for (k = 0; k < hfftl; k++) ccs2->data[k] *=
	    MIN(20.0, 1.0 + mag * pb2->data[k] * SQUARE(f0l->data[ii]));
	dvfftturn(ccs2);
	ngg = xdvifft(ccs2, ccs2->length);
	dvreal(ngg);

	for (k = 0; k < hfftl; k++) n2sgram->data[ii][k] =
	    (fabs(ngg->data[k]) + ngg->data[k]) / 2.0 + 0.1;

	/* memory free */
	xdvfree(ffsb);
	xdvfree(ffs);
	xdvfree(ccs2);
	xdvfree(ngg);
    }

    if (msg_flag == XTRUE) printmsg(stderr, "         spectrogram reshaping\n");
    
    /* memory free */
    xdvfree(pb2);
}

/* Relative aperiodic energy estimation */
DMATRIX aperiodicpart4(DVECTOR x,	/* input speech */
		       DVECTOR f0l,	/* fundamental frequency (Hz) */
		       double fs,	/* sampling frequency (Hz) */
		       double shiftm,	/* frame shift (ms) for input F0 */
		       double intshiftm,	/* frame shift (ms) for internal processing */
		       long mm,	/* length of frequency axis */
		       XBOOL msg_flag)
{
    long k, iix, ii;
    long fftl, fftl2, hfftl, len, len2;
    double jj;
    double fr40, mean;
    double fframe, endt, tmpa, tmpb;
    double gfg1, dfd1;
    DVECTOR deltat0 = NODATA;	DVECTOR wcc = NODATA;	DVECTOR fxa = NODATA;
    DVECTOR usx = NODATA;	DVECTOR txc = NODATA;	DVECTOR rv = NODATA;
    DVECTOR rt = NODATA;	DVECTOR rt2 = NODATA;	DVECTOR xc = NODATA;
    DVECTOR f0x = NODATA;	DVECTOR mllf = NODATA;	DVECTOR xctbl = NODATA;
    DVECTOR xs = NODATA;	DVECTOR a = NODATA;	DVECTOR sms = NODATA;
    DVECTOR gg = NODATA;	DVECTOR gg2 = NODATA;	DVECTOR gfg = NODATA;
    DVECTOR gfg2 = NODATA;	DVECTOR dd = NODATA;	DVECTOR dd2 = NODATA;
    DVECTOR dfd = NODATA;	DVECTOR dfd2 = NODATA;	DVECTOR apvv = NODATA;
    DVECTOR dpvv = NODATA;
    DMATRIX dapvb = NODATA;	DMATRIX dapv = NODATA;

    /* FFT size selection to be scalable */
    fftl = (long)pow(2.0, ceil(log(6.7 * fs / 40.0) / log(2.0)));
    fftl2 = fftl / 2;
    hfftl = fftl2 + 1;
    /* nearesr frequency for 40 Hz */
    fr40 = round(40.0 / fs * (double)fftl) / (double)fftl * fs;
    /* frame update frequency (Hz) */
    fframe = 1000.0 / shiftm;

    /* memory allocation */
    deltat0 = xdvalloc(f0l->length);
    /* dynamic feature */
    for (k = 1; k < f0l->length - 1; k++)
    	deltat0->data[k] = (f0l->data[k + 1] - f0l->data[k-1]) / 2.0 * fframe;
    deltat0->data[0] = (f0l->data[1] - f0l->data[0]) * fframe;
    deltat0->data[f0l->length - 1] = (f0l->data[f0l->length - 1] - f0l->data[f0l->length - 2]) * fframe;

    /* window design for 40 Hz */
    wcc = xdvwin_aperiodicpart4(fftl, fs, fr40);

    /* selector design */
    fxa = xdvalloc(mm);
    for (k = 0; k < mm; k++)
	fxa->data[k] = (double)k / (double)(mm - 1) * fs / 2.0;
    
    /* analysis for each frame */
    rv = xdvrandn(fftl * 2);	dvscoper(rv, "*", 0.0001);
    /* up sampling */
    usx = interp(x, 2);
    xc = xdvalloc(usx->length + rv->length * 2);
    dvpaste(xc, rv, 0, rv->length, 0);
    dvpaste(xc, usx, rv->length, usx->length, 0);
    xdvfree(rv);
    rv = xdvrandn(fftl * 2);	dvscoper(rv, "*", 0.0001);
    dvpaste(xc, rv, usx->length + rv->length, rv->length, 0);
    /* memory free */
    xdvfree(rv);	xdvfree(usx);

    /* memory allocation */
    txc = xdvalloc(xc->length);		f0x = xdvalloc(f0l->length);
    rt = xdvalloc(wcc->length);		rt2 = xdvalloc(wcc->length);
    dapvb = xdmalloc(f0l->length, mm);
    for (k = 0; k < txc->length; k++)
	txc->data[k] = (double)(k - fftl * 2 - 1) / fs / 2.0;
    for (k = 0; k < rt->length; k++) {
	rt->data[k] = ((double)(k + 1) - (double)wcc->length / 2.0) / fs;
	rt2->data[k] = SQUARE(rt->data[k]);
    }
    for (k = 0; k < f0l->length; k++) {
	if (f0l->data[k] == 0) {
	    f0x->data[k] = 160.0;
	} else {
	    f0x->data[k] = f0l->data[k];
	}
	if (fabs(deltat0->data[k]) > 2000.0) deltat0->data[k] = 0.0;
    }

    /* window for liftering */
    mllf = xdvalloc(fftl);
    for (k = 0; k < fftl; k++) mllf->data[k] = 1.0 / (1.0 + exp(400.0 * (fabs((double)(k + 1 - fftl2) / fs) - 0.035)));
    dvfftshift(mllf);

    /* memory allocation */
    xctbl = xdvalloc(rt->length);
    gg = xdvalloc(fftl2 - 1);	dd = xdvalloc(fftl2 - 1);
    gfg = xdvalloc(fftl2 - 1);	dfd = xdvalloc(fftl2 - 1);

    endt = round(x->length / fs * 1000.0);
    /* loop of frame analysis */
    /*
    for (iix = 0, jj = 0.0; jj < endt; jj += intshiftm) {
    */
    for (iix = 0, jj = 0.0; jj < endt; jj += shiftm) {
	if ((ii = MAX(0, MIN(f0x->length - 1, (long)round(jj / shiftm))))
	    >= f0x->length) break; 


	/* cut signal by using linear interpolation */
	for (k = 0; k < xctbl->length; k++) 
		xctbl->data[k] = (fr40 / f0x->data[ii] * rt->data[k] -0.5 * deltat0->data[ii] / f0x->data[ii] * rt2->data[k] * SQUARE(fr40 / f0x->data[ii])) + (double)(ii + 1) * shiftm / 1000.0;

	if ((xs = interp1lq(txc, xc, xctbl)) == NODATA) {
	    printmsg(stderr, "aperiodicpart4: Failed (1)\n");
	    return NODATA;
	}

	/* multiply window */
	for (k = 0, mean = dvmean(xs); k < xs->length; k++)
	    xs->data[k] = (xs->data[k] - mean) * wcc->data[k];
	/* FFT */
	a = xdvfft(xs, fftl);
	dvabs(a);

	/* Power spectrum */
	for (k = 0; k < fftl; k++)
	    a->data[k] = 20.0 * log10(a->data[k] + 0.00000001);

	/* IFFT */
	dvifft(a);
	dvreal(a);
	/* processing such as liftering */
	sms = xdvoper(a, "*", mllf);
	/* FFT (Smoothed Power spectrum) */
	dvfft(sms);
	dvreal(sms);
	/* memory free */
	xdvfree(xs);
	xdvfree(a);

	/* extract peak and dip */
	for (k = 0, len = 0, len2 = 0; k < fftl2 - 1; k++) {
	    /* differential */
	    tmpa = sms->data[k + 1] - sms->data[k];
	    tmpb = sms->data[k + 2] - sms->data[k + 1];
	    if (tmpa > 0.0 && tmpb < 0.0) {	/* search peak */
		if (sms->data[k] != 0.0) {
		    /* frequency [Hz] */
		    gg->data[len] = (double)(k + 1) / (double)fftl * fs;
		    /* power spectrum [dB] */
		    gfg->data[len] = sms->data[k + 1];
		    len++;
		}
	    } else if (tmpa < 0.0 && tmpb > 0.0) {	/* search dip */
		if (sms->data[k] != 0.0) {
		    /* frequency [Hz] */
		    dd->data[len2] = (double)(k + 1) / (double)fftl * fs;
		    /* power spectrum [dB] */
		    dfd->data[len2] = sms->data[k + 1];
		    len2++;
		}
	    }
	}

	if (len > 0) {
	    gfg1 = gfg->data[0];
	} else {
	    gfg1 = sms->data[0];
	}
	if (len2 > 0) {
	    dfd1 = dfd->data[0];
	} else {
	    dfd1 = sms->data[0];
	}
	/* memory allocation */
	gg2 = xdvalloc(len + 2);	dd2 = xdvalloc(len2 + 2);
	gfg2 = xdvalloc(len + 2);	dfd2 = xdvalloc(len2 + 2);
	gg2->data[0] = 0.0;		dd2->data[0] = 0.0;
	gg2->data[len + 1] = fs / 2.0 * f0x->data[ii] / fr40;
	dd2->data[len2 + 1] = fs / 2.0 * f0x->data[ii] / fr40;
	gfg2->data[0] = gfg1;		dfd2->data[0] = dfd1;
	gfg2->data[len + 1] = sms->data[hfftl];
	dfd2->data[len2 + 1] = sms->data[hfftl];
	for (k = 0; k < len; k++) {
	    /* frequency * F0 / 40 [Hz] */
	    gg2->data[k + 1] = gg->data[k] * f0x->data[ii] / fr40;
	    gfg2->data[k + 1] = gfg->data[k];
	}
	for (k = 0; k < len2; k++) {
	    /* frequency * F0 / 40 [Hz] */
	    dd2->data[k + 1] = dd->data[k] * f0x->data[ii] / fr40;
	    dfd2->data[k + 1] = dfd->data[k];
	}

	/* linear interpolation (extract envelope 0--Nyquist * 40 / F0 [Hz]) */
	if ((apvv = interp1q(gg2, gfg2, fxa)) == NODATA) {
	    printmsg(stderr, "aperiodicpart4: Failed (2)\n");
	    return NODATA;
	}
	if ((dpvv = interp1q(dd2, dfd2, fxa)) == NODATA) {
	    printmsg(stderr, "aperiodicpart4: Failed (3)\n");
	    return NODATA;
	}

	/* subtraction */
	for (k = 0; k < dapvb->col; k++)
	    dapvb->data[iix][k] = dpvv->data[k] - apvv->data[k];

	iix++;
	if (iix >= f0l->length) break;

	/* memory free */
	xdvfree(gg2);	xdvfree(dd2);
	xdvfree(gfg2);	xdvfree(dfd2);
	xdvfree(apvv);	xdvfree(dpvv);
	xdvfree(sms);
    }

    /* memory allocation */
    dapv = xdmalloc(iix, mm);
    /* subtracted power spectrogram */
    for (ii = 0; ii < iix; ii++)
	for (k = 0; k < mm; k++) dapv->data[ii][k] = dapvb->data[ii][k];
    
    if (msg_flag == XTRUE)
	printmsg(stderr, "         extraction of MBE-source information\n");

    /* memory free */
    xdvfree(deltat0);	xdvfree(wcc);	xdvfree(fxa);
    xdvfree(txc);	xdvfree(xc);	xdvfree(xctbl);
    xdvfree(f0x);	xdvfree(mllf);
    xdvfree(rt);	xdvfree(rt2);
    xdvfree(gg);	xdvfree(dd);
    xdvfree(gfg);	xdvfree(dfd);
    xdmfree(dapvb);
    
    return dapv;
}

/* Resample data at a higher rate using lowpass interpolation */
DVECTOR interp(DVECTOR x, long r)
{
    long k;
    long l = 4;
    long bias;
    double a = 0.5;
    DVECTOR izx = NODATA;
    DVECTOR lpf = NODATA;
    DVECTOR usxb = NODATA;
    DVECTOR usx = NODATA;

    izx = xdvzeros(x->length * r);
    for (k = 0; k < x->length; k++) izx->data[k * r] = x->data[k];
    lpf = fir1(2 * l * r, a);
    usxb = xdvfftfilt(lpf, izx, lpf->length * 2);
    usx = xdvalloc(izx->length);
    bias = l * r;
    for (k = 0; k < izx->length; k++)
	usx->data[k] = usxb->data[k + bias] * (double)r;

    /* memory free */
    xdvfree(izx);
    xdvfree(lpf);
    xdvfree(usxb);

    return usx;
}

/* window design for 40 Hz */
DVECTOR xdvwin_aperiodicpart4(long fftl, double fs, double fr40)
{
    long k;
    long len, fftl2;
    double tt, max, sum;
    LVECTOR wbbi = NODATA;
    LVECTOR wccbi = NODATA;
    DVECTOR w = NODATA;
    DVECTOR wb = NODATA;
    DVECTOR wbb = NODATA;
    DVECTOR wcc = NODATA;
    DVECTOR wccb = NODATA;

    fftl2 = fftl / 2;
    
    /* window design for 40 Hz */
    w = xdvalloc(fftl + fftl);
    wbb = xdvalloc(fftl);
    wbbi = xlvalloc(fftl);
    /* fr40/0.2 to fr40/2 worked reasonably. But, WATCH fftl !! */
    for (k = 0, len = 0; k < fftl; k++) {
	tt = (double)(k + 1 - fftl2) / fs * fr40;
	/* gaussian window */
	w->data[k] = exp(-PI * SQUARE(tt));
	/* triungular window */
	wbb->data[k] = MAX(0.0, 1.0 - fabs(tt / 2.0));
	if (wbb->data[k] > 0.0) {
	    wbbi->data[len] = k;
	    len++;
	}
    }
    for (; k < w->length; k++) w->data[k] = 0.0;
    wb = xdvalloc(len);
    for (k = 0; k < len; k++) wb->data[k] = wbb->data[wbbi->data[k]];
    /* convolution */
    wccb = xdvfftfiltm(wb, w, wb->length * 2);
    wccbi = xlvalloc(wccb->length);
    max = dvmax(wccb, NULL);
    for (k = 0, len = 0; k < wccb->length; k++) {
	if ((wccb->data[k] /= max) > 0.00002) {
	    wccb->data[k] -= 0.00002;
	    wccbi->data[len] = k;
	    len++;
	}
    }
    wcc = xdvalloc(len);
    for (k = 0, sum = 0.0; k < len; k++) {
	wcc->data[k] = wccb->data[wccbi->data[k]];
	sum += wcc->data[k];
    }
    dvscoper(wcc, "/", sum);
    /* memory free */
    xdvfree(wb);	xdvfree(wbb);	xlvfree(wbbi);
    xdvfree(w);		 xdvfree(wccb);	xlvfree(wccbi);

    return wcc;
}

/* linear interpolation (X must be monotonically increasing) */
DVECTOR interp1l(DVECTOR x, DVECTOR y, DVECTOR xi)
{
    long ki, kx, ki2;
    LVECTOR idx = NODATA;
    DVECTOR xo = NODATA;

    xo = xdvalloc(xi->length);

    idx = ss_xqsortidx(xi->data, xi->length);
    
    for (ki = 0, ki2 = 0, kx = 0; ki < xi->length; ki++) {
	for (; kx < x->length - 1; kx++) {
	    if (x->data[kx] <= xi->data[idx->data[ki]] &&
		xi->data[idx->data[ki]] <= x->data[kx + 1]) {
		/* interpolation */
		xo->data[idx->data[ki]] = (x->data[kx + 1] - xi->data[idx->data[ki]]) * y->data[kx] + (xi->data[idx->data[ki]] - x->data[kx]) * y->data[kx + 1];
		if (x->data[kx + 1] != x->data[kx]) {
		    xo->data[idx->data[ki]] /= x->data[kx + 1] - x->data[kx];
		    ki2++;
		    break;
		} else {
		    xo->data[idx->data[ki]] = y->data[kx];
		    ki2++;
		    break;
		}
	    }
	}
    }
    /* error check */
    if (ki2 != xo->length) {
	printmsg(stderr, "Failed: Interpolation\n");
	return NODATA;
    }
    
    /* memory free */
    xlvfree(idx);

    return xo;
}

LVECTOR ss_xqsortidx(
		     double *real,
		     long length)
{
    long i;
    LVECTOR idx;
    double *cpreal;

    /* memory allocation */
    idx = xlvalloc(length);
    if ((cpreal = (double *)malloc(length * sizeof(double))) == NULL) {
	printmsg(stderr, "Can't malloc\n");
	exit(1);
    }
    for (i = 0; i < length; i++) {
	idx->data[i] = i;
	cpreal[i] = real[i];
    }

    ss_quicksort(cpreal, 0, length - 1, idx);

    /* memory free */
    xfree(cpreal);

    return idx;
}

void ss_quicksort(
		  double *array,
		  long lower,
		  long upper,
		  LVECTOR idx)
{
    long l, u;
    long ltmp;
    double bound;
    double tmp;

    bound = array[lower];
    l = lower;	/* lower */
    u = upper;	/* upper */

    do {
	while (array[l] < bound) l++;
	while (array[u] > bound) u--;
	if (l <= u) {
	    tmp = array[u];
	    array[u] = array[l];
	    array[l] = tmp;
	    ltmp = idx->data[u];
	    idx->data[u] = idx->data[l];
	    idx->data[l] = ltmp;
	    l++;
	    u--;
	}
    } while (l < u);
    if (lower < u) ss_quicksort(array, lower, u, idx);
    if (l < upper) ss_quicksort(array, l, upper, idx);
}

/* linear interpolation (X must be monotonically increasing) */
DVECTOR interp1lq(DVECTOR x, DVECTOR y, DVECTOR xi)
{
    long ki, kx, len;
    DVECTOR xo = NODATA;

    xo = xdvalloc(xi->length);

    for (ki = 0, kx = 0; ki < xi->length; ki++) {
	len = x->length;
	bisearch(x->data, 0, x->length - 1, xi->data[ki], &len, &kx);
	/* interpolation */
	xo->data[ki] = (x->data[kx + 1] - xi->data[ki]) * y->data[kx] + (xi->data[ki] - x->data[kx]) * y->data[kx + 1];
	if (x->data[kx + 1] != x->data[kx]) {
	    xo->data[ki] /= x->data[kx + 1] - x->data[kx];
	} else {
	    xo->data[ki] = y->data[kx];
	}
    }
    
    return xo;
}

void bisearch(double *x,	/* x must be monotonically increasing */
	      long fidx,	/* first index */
	      long lidx,	/* last of x */
	      double a,
	      long *len,
	      long *ansidx)
{
    long idx;

    *ansidx = fidx;
    *len = lidx - fidx + 1;
    if (*len < 2) {
	printmsg(stderr, "Erorr: binary search\n");
	exit(1);
    }
    if (*len > 2) {
	idx = *len / 2 + fidx;
	if (x[idx] > a) {
	    bisearch(x, fidx, idx, a, len, ansidx);
	} else if (x[idx] < a) {
	    bisearch(x, idx, lidx, a, len, ansidx);
	} else {
	    *ansidx = idx;
	}
    }
}


DVECTOR xdvread_cf0file(char *cf0file, double *f0shift)
{
    int tmplen = MAX_LINE - 1;
    long num, k, l, len, stpt;
    char str[MAX_LINE] = "";
    char tmp[MAX_LINE] = "";
    DVECTOR f0l = NODATA;
    FILE *fp;

    if ((fp = fopen(cf0file, "rt")) == NULL) {
	printmsg(stderr, "Can't open file: %s\n", cf0file);
	exit(1);
    }

    if (fgets(str, tmplen, fp) != NULL) {
	if (strstr(str, "# FrameShift=") == NULL &&
	    strstr(str, "#FrameShift=") == NULL) {
	    exit(1);
	} else {
	    len = strlen(str);
	    stpt = 12;
	    if (strstr(str, "# FrameShift=") != NULL) stpt++;
	    for (k = stpt, l = 0; k < len && str[k] != '\n'; k++, l++) {
		if ((str[k] <= '9' && str[k] >= '0') || str[k] == '.')
		    tmp[l] = str[k];
	    }
	    tmp[l] = '\0';
	    *f0shift = atof(tmp);
	}
    } else {
	printmsg(stderr, "Error xdvread_cf0file: file format\n");
	exit(1);
    }

    num = 0;
    while (fgets(str, tmplen, fp) != NULL) num++;

    // memory allocation
    f0l = xdvalloc(num);

    fseek(fp, 0, SEEK_SET);	fgets(str, tmplen, fp);	k = 0;
    while (fgets(str, tmplen, fp) != NULL) {
	len = strlen(str);
	for (l = 0; l < len && str[l] != ' '; l++) tmp[l] = str[l];
	tmp[l] = '\0';
	f0l->data[k] = atof(tmp);
	if (f0l->data[k] < 0.0) f0l->data[k] = 0.0;
	k++;
    }
    fclose(fp);
    if (k != num) {
	printmsg(stderr, "Error xdvread_cf0file: file format\n");
	exit(1);
    }

    return f0l;
}

DVECTOR xget_interp_f0(DVECTOR f0l, double f0shift, double shift)
{
    long k, i;
    double p, a, coef;
    DVECTOR if0l = NODATA;

    if (f0shift <= 0.0) {
	printmsg(stderr, "Error F0shift > 0.0[ms]\n");
	exit(1);
    }
    coef = shift / f0shift;
    p = 0.0;	a = 0.0;	i = 0;
    for (k = 0; i + 1 < f0l->length; k++) {
	p += coef;
	i = (long)floor(p);
	a = p - (double)i;
    }
    if (a == 0.0) k++;

    // memory allocation
    if0l = xdvalloc(k);
    p = 0.0;	a = 0.0;	i = 0;
    for (k = 0; i + 1 < f0l->length; k++) {
	if (f0l->data[i] * f0l->data[i + 1] == 0.0) {
	    if0l->data[k] = f0l->data[(long)round(p)];
	} else {
	    if0l->data[k] = (1.0 - a) * f0l->data[i] + a * f0l->data[i + 1];
	}
	p = p + coef;
	i = (long)floor(p);
	a = p - i;
    }
    if (a == 0.0) {
	if (i >= f0l->length) i = f0l->length - 1;
	if0l->data[k] = (1.0 - a) * f0l->data[i];
    }

    return if0l;
}
