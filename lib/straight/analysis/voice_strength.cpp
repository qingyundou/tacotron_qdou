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


#include "fileio.h"


DMATRIX voice_strength(DVECTOR x,	/* input waveform */
		       DVECTOR f0l,	/* F0 information */
		       double fs,	/* sampling frequency (Hz) */
		       double framem,	/* frame length (ms) */
		       double shiftm,	/* frame shift (ms) */
		       long fftl,	/* FFT length */
		       double eta)	/* fast version */
{
    long k, lt0;
    long bi, ii;		/* counter */
    long framel;		/* frame length point */
    double shiftl;		/* shift length */
    double iist;		/* counter */
    long nframe;		/* the number of frame */
    double nn, tt, nt;
    double f0 = 0.0;		/* fundamental frequency */
    double t0;			/* pitch period */
    double rmsp;
    double ave;
    double p1, p2, p, nc;
    DVECTOR xh = NODATA;
    DVECTOR rv = NODATA;
    DVECTOR xt = NODATA;	/* temporary speech */
    DVECTOR cx = NODATA;	/* cut speech */
    DVECTOR wws = NODATA;	/* window for cutting wave */
    DVECTORS filter = NODATA;	/* coefficients for filter */
    DVECTORS xts = NODATA;	/* coefficients for filter */
    DVECTORS bps = NODATA;	/* coefficients for filter */
    DMATRIX vm = NODATA;	/* smoothed spectrogram */

    /* initialize global parameter */
    framel = (long)round(framem * fs / 1000.0);
    fftl = POW2(nextpow2(MAX(fftl, framel)));

    shiftl = shiftm * fs / 1000.0;

    /* High-pass filtering using 70Hz cut-off butterworth filter */
    if ((filter = butter(6, 70.0 / fs * 2.0)) == NODATA) {
	fprintf(stderr, "straight analysis: Butterworth Filter is failed\n");
	return NODATA;
    }
    xh = xdvfilter(filter->vector[0], filter->vector[1], x);
    ave = dvsum(xh) / (double)(xh->length);
    for (ii = 0, rmsp = 0.0; ii < xh->length; ii++) {
	rmsp += pow(xh->data[ii] - ave, 2.0);
    }
    rmsp = sqrt(rmsp / (double)(xh->length - 1));
    /* memory free */
    xdvsfree(filter);

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
    //    vm = xdmzeros(nframe, 5);
    vm = xdmzeros(nframe, 4);

    // bandpass-filters
    //    bps = xdvsalloc(5);
    bps = xdvsalloc(4);
    // low-pass
    bps->vector[0] = fir1(1000 + 1, 1000.0 / (fs / 2.0));
    bps->vector[1] = fir1(1000 + 1, 2000.0 / (fs / 2.0));
    // calculating power
    for (k = 0, p1 = 0.0; k < bps->vector[0]->length; k++)
	p1 += bps->vector[0]->data[k] * bps->vector[0]->data[k];
    p1 = p1 - (0.125 - p1);
    for (k = 0, p2 = 0.0; k < bps->vector[1]->length; k++)
	p2 += bps->vector[1]->data[k] * bps->vector[1]->data[k];
    p2 = p2 - (0.25 - p2);
    xdvfree(bps->vector[1]);
    // band-pass
    bps->vector[1] = fir1bp(1000 + 1, 1000.0 / (fs / 2.0),
			    2000.0 / (fs / 2.0));
    bps->vector[2] = fir1bp(1000 + 1, 2000.0 / (fs / 2.0),
			    4000.0 / (fs / 2.0));
    /*bps->vector[3] = fir1bp(1000 + 1, 4000.0 / (fs / 2.0),
			    6000.0 / (fs / 2.0));
    bps->vector[4] = fir1bp(1000 + 1, 6000.0 / (fs / 2.0),
			    8000.0 / (fs / 2.0));*/
    // high-pass
    bps->vector[3] = fir1(1000 + 1, 4000.0 / (fs / 2.0));
    bps->vector[3]->data[501] -= 1.0;
    // normalized power
    for (k = 0, p = 0.0; k < bps->vector[1]->length; k++)
	p += bps->vector[1]->data[k] * bps->vector[1]->data[k];
    dvscoper(bps->vector[1], "*", sqrt(p1 / p));
    for (k = 0, p = 0.0; k < bps->vector[2]->length; k++)
	p += bps->vector[2]->data[k] * bps->vector[2]->data[k];
    dvscoper(bps->vector[2], "*", sqrt(p2 / p));
    /*for (k = 0, p = 0.0; k < bps->vector[3]->length; k++)
	p += bps->vector[3]->data[k] * bps->vector[3]->data[k];
    dvscoper(bps->vector[3], "*", sqrt(p2 / p));
    for (k = 0, p = 0.0; k < bps->vector[4]->length; k++)
	p += bps->vector[4]->data[k] * bps->vector[4]->data[k];
	    dvscoper(bps->vector[4], "*", sqrt(p2 / p));*/
    /*
    writedvector_txt("gomi1flt.dat", bps->vector[0]);
    writedvector_txt("gomi2flt.dat", bps->vector[1]);
    writedvector_txt("gomi3flt.dat", bps->vector[2]);
    writedvector_txt("gomi4flt.dat", bps->vector[3]);
    writedvector_txt("gomi5flt.dat", bps->vector[4]);
    exit(1);
    FILE *fp;
    if ((fp = fopen("gomi1flt.dat", "wt")) == NULL) {
	fprintf(stderr, "can't open file: gomi1flt.dat\n");	exit(1);
    }
    cx = xdvfft(bps->vector[0], fftl);
    for (k = 0; k < cx->length; k++)
	fprintf(fp, "%f	%f\n", (double)k / (double)cx->length * 16000.0, 10.0 * log10(MAX(1.0e-12, cx->data[k] * cx->data[k] + cx->imag[k] * cx->imag[k])));
    fclose(fp);	xdvfree(cx);

    if ((fp = fopen("gomi2flt.dat", "wt")) == NULL) {
	fprintf(stderr, "can't open file: gomi2flt.dat\n");	exit(1);
    }
    cx = xdvfft(bps->vector[1], fftl);
    for (k = 0; k < cx->length; k++)
	fprintf(fp, "%f	%f\n", (double)k / (double)cx->length * 16000.0, 10.0 * log10(MAX(1.0e-12, cx->data[k] * cx->data[k] + cx->imag[k] * cx->imag[k])));
    fclose(fp);	xdvfree(cx);

    if ((fp = fopen("gomi3flt.dat", "wt")) == NULL) {
	fprintf(stderr, "can't open file: gomi3flt.dat\n");	exit(1);
    }
    cx = xdvfft(bps->vector[2], fftl);
    for (k = 0; k < cx->length; k++)
	fprintf(fp, "%f	%f\n", (double)k / (double)cx->length * 16000.0, 10.0 * log10(MAX(1.0e-12, cx->data[k] * cx->data[k] + cx->imag[k] * cx->imag[k])));
    fclose(fp);	xdvfree(cx);

    if ((fp = fopen("gomi4flt.dat", "wt")) == NULL) {
	fprintf(stderr, "can't open file: gomi4flt.dat\n");	exit(1);
    }
    cx = xdvfft(bps->vector[3], fftl);
    for (k = 0; k < cx->length; k++)
	fprintf(fp, "%f	%f\n", (double)k / (double)cx->length * 16000.0, 10.0 * log10(MAX(1.0e-12, cx->data[k] * cx->data[k] + cx->imag[k] * cx->imag[k])));
    fclose(fp);	xdvfree(cx);

    if ((fp = fopen("gomi5flt.dat", "wt")) == NULL) {
	fprintf(stderr, "can't open file: gomi5flt.dat\n");	exit(1);
    }
    cx = xdvfft(bps->vector[4], fftl);
    for (k = 0; k < cx->length; k++)
	fprintf(fp, "%f	%f\n", (double)k / (double)cx->length * 16000.0, 10.0 * log10(MAX(1.0e-12, cx->data[k] * cx->data[k] + cx->imag[k] * cx->imag[k])));
    fclose(fp);	xdvfree(cx);
    exit(1);
    */

    /* filtering */
    xts = xdvsalloc(4);
    for (bi = 0; bi < 4; bi++)
	xts->vector[bi] = xdvfftfiltm(bps->vector[bi], xt, 2048);

    /* loop of frame analysis */
    for (ii = 0, iist = 0.0; ii < nframe; ii++) {
	/* calculate current f0 */
	f0 = (f0l->data[ii] <= 0.0 ? UNVOICED_F0 : f0l->data[ii]);
	t0 = fs / f0;

	/* get pitch synchronous window */
	//wws = sb_xgetsinglewin2(t0 * 2.0, framel, eta);
	wws = sb_xgetsinglewin2(160.0, framel, eta);

	for (bi = 0; bi < 4; bi++) {
	    /* cut data of xt */
	    cx = sb_xcutsig(xts->vector[bi], (long)round(iist), framel);

	    // multiply window
	    dvoper(cx, "*", wws);

	    // calculate normalized correlation coefficient
	    lt0 = (long)t0;
	    for (k = 0, nn = 0.0, tt = 0.0, nt = 0.0; k < cx->length; k++) {
		nn += cx->data[k] * cx->data[k];
		if (k + lt0 < cx->length) {
		    tt += cx->data[k + lt0] * cx->data[k + lt0];
		    nt += cx->data[k] * cx->data[k + lt0];
		}
	    }
	    nc = nt / sqrt(nn * tt);
	    lt0 = (long)t0 + 1;
	    for (k = 0, nn = 0.0, tt = 0.0, nt = 0.0; k < cx->length; k++) {
		nn += cx->data[k] * cx->data[k];
		if (k + lt0 < cx->length) {
		    tt += cx->data[k + lt0] * cx->data[k + lt0];
		    nt += cx->data[k] * cx->data[k + lt0];
		}
	    }
	    vm->data[ii][bi] = MAX(nc, nt / sqrt(nn * tt));

	    xdvfree(cx);
	}
	/* memory free */
	xdvfree(wws);
	iist += shiftl;
    }

    /* memory free */
    xdvfree(xt);
    xdvsfree(xts);
    xdvsfree(bps);

    return vm;
}
