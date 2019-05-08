/*
 *	straight_body.c : straight analysis program
 *
 *	1996/12/24	coded by K.Toyama
 *	1996/12/28	modified by H.Banno
 *	1997/3/12	version 3.0 by H.Banno
 *	2001/2/10	straight_body_c03
 *			V30k18 (matlab)	  by T. Toda
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
#include "fft.h"
#include "filter.h"

#include "tempo_sub.h"
#include "straight_body_sub.h"

#include "window.h"

/*
 *	straight analysis function
 *		(correspond to straightBodyC03.m)
 */
DMATRIX straight_body_c03(DVECTOR x,	/* input waveform */
			  DVECTOR f0l,	/* F0 information */
			  double fs,	/* sampling frequency (Hz) */
			  double framem,	/* frame length (ms) */
			  double shiftm,	/* frame shift (ms) */
			  long fftl,	/* FFT length */
			  double eta,
			  double pc,	/* power constant to represent 
				   nonlinearity to perserve Ex */
			  XBOOL fast_flag,	/* fast version */
			  XBOOL msg_flag)
{
    long ii, jj;		/* counter */
    long hfftl;			/* half of fft length */
    long framel;		/* frame length point */
    double shiftl;		/* shift length */
    double iist;		/* counter */
    long nframe;		/* the number of frame */
    long ncw;
    long lbb;
    double f0 = 0.0;		/* fundamental frequency */
    double t0;			/* pitch period */
    double bcf;
    double rmsp;
    double ave;
    double lowestf0;
    double ttlv;
    double ttlv2;
    double pc2;
    XBOOL smoothinid_flag;	/* flag of smoothing in time domain */
    DVECTOR xh = NODATA;
    DVECTOR xh2 = NODATA;
    DVECTOR xhh = NODATA;
    DVECTOR rv = NODATA;
    DVECTOR ovc = NODATA;
    DVECTOR f0x = NODATA;
    DVECTOR pwc = NODATA;
    DVECTOR sumvec = NODATA;
    DVECTOR cfv = NODATA;
    DVECTOR muv = NODATA;
    DVECTOR etatbl = NODATA;
    DVECTOR bcftbl = NODATA;
    DVECTOR xt = NODATA;	/* temporary speech */
    DVECTOR cx = NODATA;	/* cut speech */
    DVECTOR wws = NODATA;	/* window for cutting wave */
    DVECTOR wwd = NODATA;	/* window for cutting wave */
    DVECTOR pw = NODATA;	/* fft power */
    DVECTOR pws = NODATA;	/* fft power */
    DVECTOR pwd = NODATA;	/* fft power */
    DVECTOR spw = NODATA;	/* fft smoothed power */
//    DVECTORS filter = NODATA;	/* coefficients for filter */
    DMATRIX n2sgram = NODATA;	/* smoothed spectrogram */

    /* initialize global parameter */
    framel = (long)round(framem * fs / 1000.0);
    fftl = POW2(nextpow2(MAX(fftl, framel)));
    hfftl = fftl / 2 + 1;

    if (fast_flag == XFALSE) {
	pc2 = 2.0;
	smoothinid_flag = XTRUE;
    } else {
	pc2 = 1.0;
	smoothinid_flag = XFALSE;
    }
    
    /* error check */
    /* if (shiftm > 1.5){ */ /* Modified by Junichi */
    if (shiftm > 5.5){
	if (msg_flag == XTRUE) {
	    fprintf(stderr, "straight analysis: frame shift must be small enough\n");
	    fprintf(stderr, "straight analysis: less than 1 ms is recommended\n");
	    fprintf(stderr, "straight analysis: temporal smoothing will be disabled\n");
	}
	smoothinid_flag = XFALSE;
    }
    shiftl = shiftm * fs / 1000.0;

    /* High-pass filtering using 70Hz cut-off butterworth filter
    if ((filter = butter(6, 70.0 / fs * 2.0)) == NODATA) {
	fprintf(stderr, "straight analysis: Butterworth Filter is failed\n");
	return NODATA;
    }
    xh = xdvfilter(filter->vector[0], filter->vector[1], x);
    // memory free
    xdvsfree(filter);
    */
    xh = xdvclone(x);
    cleaninglownoise(xh, fs, 70.0);

    ave = dvsum(xh) / (double)(xh->length);
    for (ii = 0, rmsp = 0.0; ii < xh->length; ii++) {
	rmsp += pow(xh->data[ii] - ave, 2.0);
    }
    rmsp = sqrt(rmsp / (double)(xh->length - 1));
    /* convert signal for processing */
    rv = xdvrandn(framel / 2);	dvscoper(rv, "*", rmsp / 4000.0);
    xt = xdvalloc(xh->length + framel / 2 + framel);
    dvpaste(xt, rv, 0, rv->length, 0);
    dvpaste(xt, xh, framel / 2, xh->length, 0);
    xdvfree(rv);
    rv = xdvrandn(framel);	dvscoper(rv, "*", rmsp / 4000.0);
    dvpaste(xt, rv, xh->length + framel / 2, rv->length, 0);
    /* memory free */
    xdvfree(xh);
    xdvfree(rv);

    /* get number of frame */
    nframe = MIN(f0l->length, (long)round((double)x->length / shiftl));

    /* memory allocation */
    n2sgram = xdmzeros(nframe, hfftl);
    f0x = xdvalloc(f0l->length);
    
    /* Optimum blending table for interference free spec. 1.4 & 0.43 */
    cfv = xdvalloc(7);	muv = xdvalloc(7);    etatbl = xdvalloc(7);
    cfv->data[0] = 1.03; cfv->data[1] = 0.83; cfv->data[2] = 0.67;
    cfv->data[3] = 0.54; cfv->data[4] = 0.43; cfv->data[5] = 0.343;
    cfv->data[6] = 0.2695;
    muv->data[0] = 1.0;  muv->data[1] = 1.1;  muv->data[2] = 1.2;
    muv->data[3] = 1.3;  muv->data[4] = 1.4;  muv->data[5] = 1.5;
    muv->data[6] = 1.6;
    for (ii = 0; ii < 7; ii++) etatbl->data[ii] = eta;
    /* linear interpolation (matlab version: spline) */
    bcftbl = interp1q(muv, cfv, etatbl);
    bcf = bcftbl->data[3];
    /* memory free */
    xdvfree(cfv);	xdvfree(muv);
    xdvfree(etatbl);	xdvfree(bcftbl);

    ovc = optimumsmoothing(eta, pc);

    /* loop of smoothing in frequency domain */
    for (ii = 0, iist = 0.0; ii < nframe; ii++) {
	/* calculate current f0 */
	f0 = (f0l->data[ii] <= 0.0 ? UNVOICED_F0 : f0l->data[ii]);
	f0x->data[ii] = f0;
	t0 = fs / f0;

	/* cut data of xt */
	cx = sb_xcutsig(xt, (long)round(iist), framel);

	/* get pitch synchronous window */
	wws = sb_xgetsinglewin2(t0, framel, eta);
	wwd = sb_xgetdoublewin2(wws, t0, framel, bcf);

	/* calculate fft power spectrum */
	pws = sb_xgetfftpow(cx, wws, fftl, 2.0);
	pwd = sb_xgetfftpow(cx, wwd, fftl, 2.0);

	/* calculate power spectrum using double window */
	pw = sb_xgetdbfftpow(pws, pwd, pc);

	/* smoothing of frequency domain */
	spw = sb_xsmoothfreq_c03(pw, t0, fftl, ovc);

	/* (pc2/pc) power of spectrum */
	dvscoper(spw, "^", pc2 / pc);

	/* copy power spectrum to ii-th row of matrix */
	dmcopyrow(n2sgram, ii, spw);
	//dvscoper(pw, "^", pc2 / pc);
	//dvscoper(pwd, "^", 0.5);
	//dmcopyrow(n2sgram, ii, pwd);

	/* memory free */
	xdvfree(cx);
	xdvfree(wws);
	xdvfree(wwd);
	xdvfree(pws);
	xdvfree(pwd);
	xdvfree(pw);
	xdvfree(spw);

	iist += shiftl;
    }

    if (msg_flag == XTRUE) fprintf(stderr, "         frequency smoothing\n");

    /* memory free */
    xdvfree(ovc);
    xdvfree(xt);

    /* smoothing in time domain */
    if (smoothinid_flag == XTRUE) {
	for (; ii < f0l->length; ii++) f0x->data[ii] = f0;
	lowestf0 = 40.0;
	sb_xsmoothtime_c03(n2sgram, f0x, lowestf0, shiftm);
	if (msg_flag == XTRUE) fprintf(stderr, "         temporal smoothing\n");
    }
    
    /* memory free */
    xdvfree(f0x);

    if (fast_flag == XFALSE) {
	/* High-pass filtering using 300Hz cut-off butterworth filter
	if ((filter = butter(6, 300.0 / fs * 2.0)) == NODATA) {
	    fprintf(stderr,
		    "straight analysis: Butterworth Filter is failed\n");
	    return NODATA;
	}
	xh2 = xdvfilter(filter->vector[0], filter->vector[1], x);
	// memory free
	xdvsfree(filter);
	*/
	xh2 = xdvclone(x);
	cleaninglownoise(xh2, fs, 300.0);

	/* High-pass filter using 3000Hz cut-off butterworth filter
	if ((filter = butter(6, 3000.0 / fs * 2.0)) == NODATA) {
	    fprintf(stderr,
		    "straight analysis: Butterworth Filter is failed\n");
	    return NODATA;
	}
	xhh = xdvfilter(filter->vector[0], filter->vector[1], x);
	// memory free
	xdvsfree(filter);
	*/
	xhh = xdvclone(x);
	cleaninglownoise(xhh, fs, 3000.0);
	
	/* Dirty hack for controling time constant in unvoiced part analysis */
	ncw = (long)round(2.0 * fs / 1000.0);
	lbb = (long)round(300.0 / fs * (double)fftl);
	sumvec = xdvalloc(n2sgram->row);
	/* calculate power */
	for (ii = 0, ttlv = 0.0, ttlv2 = 0.0; ii < n2sgram->row; ii++) {
	    for (jj = 0, sumvec->data[ii] = 0.0; jj < n2sgram->col; jj++) {
		sumvec->data[ii] += n2sgram->data[ii][jj];
		if (jj > lbb - 2) ttlv2 += n2sgram->data[ii][jj];
	    }
	    ttlv += sumvec->data[ii];
	}

	/* calculate time constant controller */
	pwc = sb_xtconstuv_c03(n2sgram, xh2, xhh, sumvec, ttlv, ttlv2, fs,
			       shiftm, ncw);

	/* Shaping amplitude envelope */
	for (ii = 0; ii < n2sgram->row; ii++) {
	    if (f0l->data[ii] == 0) {
		for (jj = 0; jj < n2sgram->col; jj++) {
		    /* floating error */
		    if (sumvec->data[ii] <= 0.0) {
			n2sgram->data[ii][jj] *= 0.0;
		    } else {
			n2sgram->data[ii][jj] *= pwc->data[ii] / sumvec->data[ii];
		    }
		    n2sgram->data[ii][jj] += 0.0000000001;
		    /* amplitude spectrum */
		    n2sgram->data[ii][jj] = sqrt(fabs(n2sgram->data[ii][jj]));
		}
	    } else {
		for (jj = 0; jj < n2sgram->col; jj++) {
		    n2sgram->data[ii][jj] += 0.0000000001;
		    /* amplitude spectrum */
		    n2sgram->data[ii][jj] = sqrt(fabs(n2sgram->data[ii][jj]));
		}
	    }
	}
	if (msg_flag == XTRUE) fprintf(stderr, "         unvoiced part shaping\n");

	/* memory free */
	xdvfree(xh2);
	xdvfree(xhh);
	xdvfree(pwc);
	xdvfree(sumvec);
    }

    return n2sgram;
}
