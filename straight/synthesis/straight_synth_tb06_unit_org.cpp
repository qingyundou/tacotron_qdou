/*
 *	straight_synth.c : straight synthesis program
 *
 * 	1996/12/25	coded by T.Doi & H.Banno
 *	2001/2/12	straight_synth_tb06
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
#include "voperate.h"
#include "fft.h"
#include "filter.h"
#include "window.h"
#include "matrix.h"

#include "straight_sub.h"
#include "straight_body_sub.h"
#include "straight_synth_sub.h"
#include "straight_vconv_sub.h"

/* STRAIGHT Synthesis Using Parameters Generated from HMM */
DVECTOR straight_synth_tb06(DMATRIX n2sgram,	/* amplitude spectrogram */
			    DVECTOR f0l,	/* pitch pattern (Hz) */
			    DVECTOR f0var,	/*  expected F0 variation */
			    double fs,		/* sampling freqnency (Hz) */
			    double shiftm,	/* frame shift (ms) */
			    double pconv,	/* pitch stretch factor */
			    double fconv,	/* freqnency stretch factor */
			    double sconv,	/* time stretch factor */
			    double gdbw,	/* band width of group delay (Hz) */
			    double delsp,	/* standard deviation of group delay (ms)*/
			    double cornf,	/* lower corner frequency for phase randomization (Hz) */
			    double delfrac,	/* ratio of standard deviation of group delay in terms of F0 */
			    XBOOL fr_flag,	/* using fractional pitch */
			    XBOOL zp_flag,	/* using zero phase */
			    XBOOL rp_flag,	/* using random phase */
			    XBOOL df_flag)	/* using proportional group delay */
{
    long k, l, nii, njj, lii;
    long num, nsyn, nidx, maxidx;
    long fftl, fftl2, hfftl;
    double lowcutf, value;
    double ii, iin, iix, idx, idxo, ipos;
    double nf0 = 0.0, f0 =  0.0;
    double frt, trbw, dmx;
    LVECTOR idcv = NODATA;
    DVECTOR sy = NODATA;	DVECTOR sy2 = NODATA;
    DVECTOR wlcut = NODATA;	DVECTOR wlcutfric = NODATA;
    DVECTOR fgw = NODATA;	DVECTOR han = NODATA;
    DVECTOR rho = NODATA;	DVECTOR rho2 = NODATA;
    DVECTOR lft = NODATA;	DVECTOR ww = NODATA;
    DVECTOR ccp = NODATA;	DVECTOR spc = NODATA;
    DVECTOR mxv = NODATA;	DVECTOR mxn = NODATA;
    DVECTOR tx = NODATA;	DVECTOR tnx = NODATA;
    DVECTOR rx = NODATA;
    XBOOL end_flag = XFALSE;

    /* initialize */
    nii = n2sgram->col;
    njj = n2sgram->row;
    fftl = 2 * (nii - 1);
    fftl2 = fftl / 2;
    hfftl = fftl2 + 1;
    trbw = 300.0;

    /* error check */
    if (fftl != POW2(nextpow2(fftl))) {
	fprintf(stderr, "straight synth: wrong format of analysis data\n");
	return NODATA;
    }
    if (fs == 0.0 || shiftm == 0.0 || pconv == 0.0 || fconv == 0.0 ||
	sconv == 0.0) {
	fprintf(stderr, "straight synth: wrong parameter value\n");
	return NODATA;
    }

    /* memory allocation */
    nsyn = (long)round(sconv * (double)njj * shiftm * fs / 1000.0 +
		       3.0 * (double)fftl + 1.0);
    sy = xdvzeros(nsyn);

    /* make frequency stretch table */
    idcv = xlvinit(0, 1, fftl2);
    lvscoper(idcv, "/", fconv);
    lvscmin(idcv, fftl2);

    /* memory allocation */
    mxv = xdvalloc(f0var->length);	mxn = xdvalloc(f0var->length);
    for (k = 0; k < f0var->length; k++) {
	mxv->data[k] = sqrt(0.25 / (f0var->data[k] + 0.25));
	mxn->data[k] = sqrt(1.0 - 0.25 / (f0var->data[k] + 0.25));
    }

    /* shaping for low-frequency noize supression */
    for (k = 0, num = 0, value = 0.0; k < f0l->length; k++) {
	if (mxv->data[k] > 0.8 && f0l->data[k] > 0.0) {
	    value += f0l->data[k];
	    num++;
	}
    }
    if (num != 0) {
	value /= (double)num;
    } else {
	value = 70.0;
    }
    lowcutf = value * 0.7 * pconv;
    /* memory allocation */
    wlcut = xdvalloc(hfftl);

    /* smoothing window for group delay in frequency domain */
    fgw = ss_xfgrpdlywin(fs, gdbw, fftl);

    /* group delay weighting function */
    rho = ss_xgdweight(fs, trbw, cornf, fftl);
    rho2 = xdvalloc(rho->length);

    /* make lifter and time window */
    han = xdvhanning(fftl);
    lft = xdvalloc(fftl);
    ww = xdvalloc(fftl);
    for (k = 0; k < fftl; k++) {
	lft->data[k] = 1.0 / (1.0 + exp((han->data[k] - 0.5) * 60.0));
	ww->data[k] = 1.0 / (1.0 + exp(-(han->data[k] - 0.3) * 23.0));
    }
    /* memory free */
    xdvfree(han);

    for (k = 0, dmx = 0.0; k < n2sgram->row; k++) {
	for (l = 0; l < n2sgram->col; l++) {
	    if (dmx < n2sgram->data[k][l]) dmx = n2sgram->data[k][l];
	}
    }

    maxidx = nsyn - fftl - 11;
    
    iin = 0.0;
    idx = 0.0;	end_flag = XFALSE;
    while ((long)idx < maxidx && (long)ceil(iin) < f0l->length - 1) {
	iix = (idx + 1.0) * 1000.0 / fs / shiftm / sconv;
	ii = MIN(MIN(MAX(0.0, iix), (double)njj - 1.0),
		 (double)f0l->length - 1.0);
	lii = (long)round(ii);

	// error check
	if (end_flag == XTRUE) break;
	if (lii == n2sgram->row - 1) end_flag = XTRUE;

	f0 = MAX(40.0, f0l->data[lii]);
	f0 = f0 * pconv;

	for (k = 0; k < hfftl; k++) wlcut->data[k] = 1.0 / (1.0 + exp(-10.0 *((double)k / (double)fftl * fs - f0 * 0.7) / f0));

	/* calculate cepstrum */
	ccp = ss_xextractcep_tb06(n2sgram, lii, idcv, fftl, wlcut,
				  dmx / 1000000.0);

	if (zp_flag != XTRUE) ss_ceptompc(ccp, fftl);

	/* liftering */
	dvoper(ccp, "*", lft);

	/* calculate spectrum */
	spc = ss_xceptospec(ccp, NULL, fftl);

	nidx = (long)round(idx);

	nf0 = fs / f0;
	if (fr_flag == XTRUE) {
	    frt = idx - (double)nidx;
	} else {
	    frt = 0.0;
	}

	/* design apf for fractional pitch */
	ss_fractpitchspec(spc, frt, fftl);

	/* design apf for random phase */
	if (rp_flag == XTRUE) {
	    if (df_flag == XTRUE) delsp = delfrac * 1000.0 / f0;
	    for (k = 0; k < rho->length; k++) rho2->data[k] =
		mxn->data[lii] + (1 - mxn->data[lii]) * rho->data[k];
	    ss_randomspec(spc, fgw, rho2, fs, gdbw, delsp, fftl);
	}
	dvscoper(spc, "*", mxv->data[lii]);

	/* get waveform */
	tx = ss_xspectowave(spc, fftl);
	/* multiply time window */
	dvoper(tx, "*", ww);
	dvscoper(tx, "*", sqrt(nf0));

	/* overlap add (nidx + 1: correspond to matlab) */
	dvpaste(sy, tx, nidx + 1, tx->length, 1);

	/* memory free */
	xdvfree(ccp);
	xdvfree(spc);
	xdvfree(tx);

	idx += nf0;
	iin = MIN((double)f0l->length - 1.0,
		  (idx + 1.0) * 1000.0 / fs / shiftm / sconv);

	if (mxv->data[lii] < 0.8 && mxv->data[(long)round(iin)] > 0.8) {
	    idxo = idx;
	    for (k = lii, ipos = -1.0; k <= (long)round(iin); k++) {
		if (mxv->data[k] > 0.8) {
		    ipos = k - lii + ii;
		    break;
		}
	    }
	    if (ipos != -1.0) idx = MAX(idxo - nf0 + 1.0, ipos * fs / 1000.0 * shiftm * sconv) - 1.0;
	}
    }

    /* memory free */
    xdvfree(wlcut);
    xdvfree(fgw);

    /* memory allocation */
    wlcutfric = xdvalloc(hfftl);
    for (k = 0; k < hfftl; k++) wlcutfric->data[k] = 1.0 / (1.0 + exp(-14.0 * ((double)k / (double)fftl * fs - lowcutf) / lowcutf));

    ii = 0.0;
    idx = 0.0;
    f0 = 1000.0;
    maxidx = nsyn - fftl - 1;	end_flag = XFALSE;
    while ((long)idx < maxidx && (long)ii < f0l->length - 1) {
	ii = round(MIN(MIN((double)f0l->length - 1.0, njj - 1.0),
		       (idx + 1.0) * 1000.0/ fs / shiftm / sconv));
	lii = (long)round(ii);
	nidx = (long)round(idx);
	
	// error check
	if (end_flag == XTRUE) break;
	if (lii == n2sgram->row - 1) end_flag = XTRUE;

	if (mxn->data[lii] > 0.03) {
	    /* calculate cepstrum */
	    ccp = ss_xextractcep_tb06(n2sgram, lii, idcv, fftl, wlcutfric,
				      dmx / 100000.0);

	    if (zp_flag != XTRUE) ss_ceptompc(ccp, fftl);
	    
	    /* liftering */
	    dvoper(ccp, "*", lft);

	    /* calculate spectrum */
	    spc = ss_xceptospec(ccp, NULL, fftl);

	    nf0 = fs / f0;
	    
	    /* get waveform */
	    tx = ss_xspectowave(spc, fftl);

	    /* noise-excitation */
	    rx = xdvrandn((long)round(nf0));

	    /* convolution */
	    tnx = xdvfftfiltm(rx, tx, rx->length * 2);
	    /* multiply time window */
	    dvoper(tnx, "*", ww);
	    
	    /* overlap add (nidx + 1: correspond to matlab) */
	    dvpaste(sy, tnx, nidx + 1, tnx->length, 1);

	    /* memory free */
	    xdvfree(ccp);
	    xdvfree(spc);
	    xdvfree(tx);
	    xdvfree(tnx);
	    xdvfree(rx);
	}

	idx += nf0;
	ii = MIN((double)f0l->length - 1.0,
		 (idx + 1.0) * 1000.0 / fs / shiftm / sconv);
    }
    
    /* memory free */
    xdvfree(wlcutfric);
    xdvfree(rho2);
    xdvfree(mxn);
    xdvfree(mxv);

    sy2 = xdvcut(sy, fftl2,
		 (long)((double)njj * shiftm * fs / 1000.0 * sconv));

    ss_waveampcheck_tb06(sy2, fs, 15.0);

    /* memory free */
    xlvfree(idcv);
    xdvfree(rho);
    xdvfree(lft);
    xdvfree(ww);
    xdvfree(sy);
    ss_xfree_sub();

    return sy2;
}

double sigmoid(double x, double a, double b)
{
  double s0, s1;
  s0 = 1.0 / (1.0 + exp(-1.0 * a * (0.0 - b)));
  s1 = 1.0 / (1.0 + exp(-1.0 * a * (1.0 - b)));
  return (1.0 / (1.0 + exp(-1.0 * a * (x - b))) - s0) / (s1 - s0);
}

/* STRAIGHT Synthesis Using Parameters Generated from HMM */
/* Graded excitation */
DVECTOR straight_synth_tb06ca(DMATRIX n2sgram,	/* amplitude spectrogram */
			      DVECTOR f0l,	/* pitch pattern (Hz) */
			      double fs,	/* sampling freqnency (Hz) */
			      double shiftm,	/* frame shift (ms) */
			      double pconv,	/* pitch stretch factor */
			      double fconv,	/* freqnency stretch factor */
			      double sconv,	/* time stretch factor */
			      double gdbw,	/* band width of group delay (Hz) */
			      double delsp,	/* standard deviation of group delay (ms)*/
			      double cornf,	/* lower corner frequency for phase randomization (Hz) */
			      double delfrac,	/* ratio of standard deviation of group delay in terms of F0 */
			      DMATRIX ap,	/* aperiodicity measure */
			      DVECTOR imap,	/* arbirtary mapping from new time (sample) to old time (frame) */
			      XBOOL fr_flag,	/* using fractional pitch */
			      XBOOL zp_flag,	/* using zero phase */
			      XBOOL rp_flag,	/* using random phase */
			      XBOOL df_flag)	/* using proportional group delay */
{
    long k, l, nii, njj, lii, ix;
    long num, nsyn, nidx, maxidx;
    long fftl, fftl2, hfftl;
    long f0len;
    double lowcutf, value;
    double ii, iin, iix, idx, idxo, ipos;
    double nf0 = 0.0, f0 = 0.0;
    double frt, trbw, dmx;
    LVECTOR idcv = NODATA;
    DVECTOR sy = NODATA;	DVECTOR sy2 = NODATA;
    DVECTOR wlcut = NODATA;	DVECTOR wlcutfric = NODATA;
    DVECTOR fgw = NODATA;	DVECTOR han = NODATA;
    DVECTOR rho = NODATA;	DVECTOR ccp = NODATA;
    DVECTOR lft = NODATA;	DVECTOR ww = NODATA;
    DVECTOR spc = NODATA;    	DVECTOR spc2 = NODATA;
    DVECTOR tx = NODATA;	DVECTOR tx2 = NODATA;
    DVECTOR tnx = NODATA;	DVECTOR rx = NODATA;
    DVECTOR tmpimap = NODATA;	DVECTOR cpimap = NODATA;
    DVECTOR imapx = NODATA;   	DVECTOR imapy = NODATA;
    DVECTOR imapi = NODATA;	DVECTOR rmap = NODATA;
    DVECTOR wnz = NODATA;	DVECTOR wpr = NODATA;
    DVECTOR wfv = NODATA;
    XBOOL end_flag = XTRUE;

    /* initialize */
    nii = n2sgram->col;
    njj = n2sgram->row;
    njj = MIN(njj, f0l->length);
    f0len = njj;
    fftl = 2 * (nii - 1);
    fftl2 = fftl / 2;
    hfftl = fftl2 + 1;
    trbw = 300.0;

    /* error check */
    if (fftl != POW2(nextpow2(fftl))) {
	fprintf(stderr, "straight synth: wrong format of analysis data\n");
	return NODATA;
    }
    if (fs == 0.0 || shiftm == 0.0 || pconv == 0.0 || fconv == 0.0 ||
	sconv == 0.0) {
	fprintf(stderr, "straight synth: wrong parameter value\n");
	return NODATA;
    }

    if (imap != NODATA) {
	/* memory allocation */
	nsyn = imap->length + 3 * fftl;
	sy = xdvzeros(nsyn);
	cpimap = xdvalloc(imap->length + (long)round(fs * 0.2));
	for (k = 0, ix = -1; k < imap->length; k++) {
	    cpimap->data[k] = imap->data[k];
	    if (ix == -1) if (imap->data[k] >= f0len - 1) ix = k;
	}
    } else {
	/* memory allocation */
	nsyn = (long)round(sconv * (double)njj * shiftm * fs / 1000.0 +
			   3.0 * (double)fftl + 1.0);
	sy = xdvzeros(nsyn);
	tmpimap = xdvalloc(sy->length);
	for (k = 0; k < tmpimap->length; k++) tmpimap->data[k] =
	    MIN(f0len - 1, (double)k * 1000.0 / fs / shiftm / sconv);
	cpimap = xdvalloc(tmpimap->length + (long)round(fs * 0.2));
	for (k = 0, ix = -1; k < tmpimap->length; k++) {
	    cpimap->data[k] = tmpimap->data[k];
	    if (ix == -1) if (tmpimap->data[k] >= f0len - 1) ix = k;
	}
	/* memory free */
	xdvfree(tmpimap);
    }
    if (ix == -1) ix = k;
    for (; k < cpimap->length; k++) cpimap->data[k] = (double)(f0len - 1);
    imapx = xdvalloc(ix + 1);	imapy = xdvalloc(ix + 1);
    for (k = 0; k < imapx->length; k++) {
	imapx->data[k] = cpimap->data[k];
	imapy->data[k] = (double)k;
    }
    imapi = xdvalloc(f0len);
    for (k = 0; k < imapi->length; k++) imapi->data[k] = (double)k;
    if ((rmap = interp1lq(imapx, imapy, imapi)) == NODATA) {
	fprintf(stderr, "Error: straight_synth_tb06ca\n");
	return NODATA;
    }
    /* memory free */
    xdvfree(imapx);
    xdvfree(imapy);
    xdvfree(imapi);

    /* make frequency stretch table */
    idcv = xlvinit(0, 1, fftl2);
    lvscoper(idcv, "/", fconv);
    lvscmin(idcv, fftl2);

    /* shaping for low-frequency noize supression */
    for (k = 0, num = 0, value = 0.0; k < f0len; k++) {
	if (f0l->data[k] > 0.0) {
	    value += f0l->data[k];
	    num++;
	}
    }
    if (num != 0) {
	value /= (double)num;
    } else {
	value = 70.0;
    }
    lowcutf = value * 0.7 * pconv;
    /* memory allocation */
    wlcut = xdvalloc(hfftl);

    /* smoothing window for group delay in frequency domain */
    fgw = ss_xfgrpdlywin(fs, gdbw, fftl);

    /* group delay weighting function */
    rho = ss_xgdweight(fs, trbw, cornf, fftl);

    /* make lifter and time window */
    han = xdvhanning(fftl);
    lft = xdvalloc(fftl);
    ww = xdvalloc(fftl);
    for (k = 0; k < fftl; k++) {
	lft->data[k] = 1.0 / (1.0 + exp((han->data[k] - 0.5) * 60.0));
	ww->data[k] = 1.0 / (1.0 + exp(-(han->data[k] - 0.3) * 23.0));
    }
    /* memory free */
    xdvfree(han);

    for (k = 0, dmx = 0.0; k < n2sgram->row; k++) {
	for (l = 0; l < n2sgram->col; l++) {
	    if (dmx < n2sgram->data[k][l]) dmx = n2sgram->data[k][l];
	}
    }

    /* memory allocation */
    wnz = xdvalloc(fftl);
    wpr = xdvalloc(fftl);

    maxidx = nsyn - fftl - 11;
    
    iin = 0.0;
    idx = 0.0;	end_flag = XFALSE;
    while ((long)idx < maxidx && (long)ceil(iin) < f0len - 1) {
	iix = round(cpimap->data[(long)round(idx)]);
	ii = MIN(MIN(MAX(0.0, iix), (double)njj - 1.0),
		 (double)f0len - 1.0);
	lii = (long)round(ii);

	// error check
	if (end_flag == XTRUE) break;
	if (lii == n2sgram->row - 1) end_flag = XTRUE;

	f0 = MAX(40.0, f0l->data[lii]);
	f0 = f0 * pconv;
	nf0 = fs / f0;

	if (f0l->data[lii] > 0.0) {
	    for (k = 0; k < hfftl; k++) wlcut->data[k] = 1.0 / (1.0 + exp(-10.0 *((double)k / (double)fftl * fs - f0 * 0.7) / f0));
	
	    /* calculate cepstrum */
	    ccp = ss_xextractcep_tb06(n2sgram, lii, idcv, fftl, wlcut,
				      dmx / 1000000.0);

	    if (zp_flag != XTRUE) ss_ceptompc(ccp, fftl);

	    /* liftering */
	    dvoper(ccp, "*", lft);

	    /* calculate spectrum */
	    spc = ss_xceptospec(ccp, NULL, fftl);
	
	    nidx = (long)round(idx);

	    if (fr_flag == XTRUE) {
		frt = idx - (double)nidx;
	    } else {
		frt = 0.0;
	    }

	    /* design apf for fractional pitch */
	    ss_fractpitchspec(spc, frt, fftl);

	    /* design apf for random phase */
	    if (rp_flag == XTRUE) {
		if (df_flag == XTRUE) delsp = delfrac * 1000.0 / f0;
		ss_randomspec(spc, fgw, rho, fs, gdbw, delsp, fftl);
	    }

	    /* weighting for graded excitation */
	    wnz->data[0] = pow(10.0, ap->data[lii][idcv->data[0]] / 20.0);
	    wpr->data[0] = sqrt(MAX(0.0, 1.0 - SQUARE(wnz->data[0])));
	    for (k = 1; k < ap->col - 1; k++) {
		wnz->data[k] = MIN(1.0, sigmoid(pow(10.0, ap->data[lii][idcv->data[k]] / 20.0), 6.0, 0.25));
		wpr->data[k] = sqrt(MAX(0.0, 1.0 - SQUARE(wnz->data[k])));
		wnz->data[fftl - k] = wnz->data[k];
		wpr->data[fftl - k] = wpr->data[k];
	    }
	    wnz->data[k] = pow(10.0, ap->data[lii][idcv->data[k]] / 20.0);
	    wpr->data[k] = sqrt(MAX(0.0, 1.0 - SQUARE(wnz->data[k])));

	    /* noise-excitation */
	    rx = xdvrandn((long)round(nf0));

	    wfv = xdvfft(rx, fftl);
	    dvoper(wfv, "*", wnz);

	    spc2 = xdvoper(spc, "*", wfv);
	    dvoper(spc, "*", wpr);

	    /* get waveform */
	    tx = ss_xspectowave(spc, fftl);
	    tx2 = ss_xspectowave(spc2, fftl);

	    /* multiply time window */
	    for (k = 0; k < tx->length; k++) {
		tx->data[k] = tx->data[k] * sqrt(nf0) + tx2->data[k];
		tx->data[k] *= ww->data[k];
	    }

	    /* overlap add (nidx + 1: correspond to matlab) */
	    dvpaste(sy, tx, nidx + 1, tx->length, 1);

	    /* memory free */
	    xdvfree(ccp);
	    xdvfree(spc);
	    xdvfree(spc2);
	    xdvfree(tx);
	    xdvfree(tx2);
	    xdvfree(wfv);
	    xdvfree(rx);
	}

	idx += nf0;
	iin = round(cpimap->data[(long)round(idx)]);

	if (f0l->data[lii] == 0.0 && f0l->data[(long)round(iin)] > 0.0) {
	    idxo = idx;
	    for (k = lii, ipos = -1.0; k <= (long)round(iin); k++) {
		if (f0l->data[k] > 0.0) {
		    ipos = k - lii + ii;
		    break;
		}
	    }
	    if (ipos != -1.0) idx = MAX(idxo - nf0 + 1.0,
					rmap->data[(long)round(ipos - 1.0)]);
	}
    }

    /* memory free */
    xdvfree(wlcut);
    xdvfree(fgw);
    xdvfree(wnz);
    xdvfree(wpr);
    xdvfree(rmap);
    
    /* memory allocation */
    wlcutfric = xdvalloc(hfftl);
    for (k = 0; k < hfftl; k++) wlcutfric->data[k] = 1.0 / (1.0 + exp(-14.0 * ((double)k / (double)fftl * fs - lowcutf) / lowcutf));

    ii = 0.0;
    idx = 0.0;
    f0 = 1000.0;
    maxidx = nsyn - fftl - 1;	end_flag = XFALSE;
    while ((long)idx < maxidx && (long)ii < f0len - 1) {
	ii = round(cpimap->data[(long)round(idx)]);
	lii = (long)round(ii);
	nidx = (long)round(idx);
	
	// error check
	if (end_flag == XTRUE) break;
	if (lii == n2sgram->row - 1) end_flag = XTRUE;

	if (f0l->data[lii] == 0.0) {
	    /* calculate cepstrum */
	    ccp = ss_xextractcep_tb06(n2sgram, lii, idcv, fftl, wlcutfric,
				      dmx / 100000.0);

	    if (zp_flag != XTRUE) ss_ceptompc(ccp, fftl);
	    
	    /* liftering */
	    dvoper(ccp, "*", lft);

	    /* calculate spectrum */
	    spc = ss_xceptospec(ccp, NULL, fftl);

	    nf0 = fs / f0;
	    
	    /* get waveform */
	    tx = ss_xspectowave(spc, fftl);

	    /* noise-excitation */
	    rx = xdvrandn((long)round(nf0));

	    /* convolution */
	    tnx = xdvfftfiltm(rx, tx, rx->length * 2);
	    /* multiply time window */
	    dvoper(tnx, "*", ww);
	    
	    /* overlap add (nidx + 1: correspond to matlab) */
	    dvpaste(sy, tnx, nidx + 1, tnx->length, 1);

	    /* memory free */
	    xdvfree(ccp);
	    xdvfree(spc);
	    xdvfree(tx);
	    xdvfree(tnx);
	    xdvfree(rx);
	}

	idx += nf0;
	ii = round(cpimap->data[(long)round(idx)]);
    }

    /* memory free */
    xdvfree(wlcutfric);

    sy2 = xdvcut(sy, fftl2, ix + 1);

    ss_waveampcheck_tb06(sy2, fs, 15.0);

    /* memory free */
    xlvfree(idcv);
    xdvfree(cpimap);
    xdvfree(rho);
    xdvfree(lft);
    xdvfree(ww);
    xdvfree(sy);
    ss_xfree_sub();

    return sy2;
}

DVECTOR xread_hmmf0(char *file)
{
    DVECTOR f0raw = NODATA;
    long len, k;
    float tmp = 0.0;
    FILE *fp;

    // open f0 file
    if ((fp = fopen(file, "rb")) == NULL) {
	printmsg(stderr, "Can't open %s file\n", file);
	return NODATA;
    }

    len = 0;
    while ((size_t)1 == fread(&tmp, sizeof(float), (size_t)1, fp)) len++;

    // memory allocation
    f0raw = xdvalloc(len);

    fseek(fp, 0, SEEK_SET);
    for (k = 0; k < len; k++) {
	if ((size_t)1 != fread(&tmp, sizeof(float), (size_t)1, fp)) {
	    printmsg(stderr, "Error xread_hmmf0: %s\n", file);
	    return NODATA;
	} else {
	    if (tmp != 0.0) {
		f0raw->data[k] = exp((double)tmp);
	    } else {
		f0raw->data[k] = 0.0;
	    }
	}
    }

    // close file
    fclose(fp);

    return f0raw;
}

DMATRIX xread_dfcep2spg(char *file, long dim, long fftl, XBOOL mel_flag,
                        XBOOL float_flag, XBOOL chpow_flag,
                        double frame, double fs, double alpha)
{
    DMATRIX spg = NODATA;
    long len, k, l;
    long hfftl;
    double spgpow, tspgpow = 0.0;
    FVECTOR tmpf = NODATA;
    DVECTOR tmpd = NODATA;
    DVECTOR vec1 = NODATA;
    DVECTOR vec2 = NODATA;
    DVECTOR vec = NODATA;
    FILE *fp;

    hfftl = fftl / 2 + 1;

    // open cepstrum file
    if ((fp = fopen(file, "rb")) == NULL) {
	printmsg(stderr, "Can't open %s file\n", file);
	return NODATA;
    }

    len = 0;
    if (float_flag == XTRUE) {
	// memory allocation
	tmpf = xfvalloc(dim);
	while ((size_t)dim ==
	       fread(tmpf->data, sizeof(float), (size_t)dim, fp)) len++;
    } else {
	// memory allocation
	tmpd = xdvalloc(dim);
	while ((size_t)dim ==
	       fread(tmpd->data, sizeof(double), (size_t)dim, fp)) len++;
	// memory free
	xdvfree(tmpd);
    }

    // memory allocation
    spg = xdmalloc(len, hfftl);
    vec1 = xdvalloc(dim);

    fseek(fp, 0, SEEK_SET);
    for (k = 0; k < len; k++) {
	if (float_flag == XTRUE) {
	    if ((size_t)dim !=
		fread(tmpf->data, sizeof(float), (size_t)dim, fp)) {
		printmsg(stderr, "Error xread_dfcep2spg: %s\n", file);
		return NODATA;
	    }
	    for (l = 0; l < dim; l++) vec1->data[l] = (double)tmpf->data[l];
	} else {
	    if ((size_t)dim != fread(vec1->data, sizeof(double), (size_t)dim,
				     fp)) {
		printmsg(stderr, "Error xread_dfcep2spg: %s\n", file);
		return NODATA;
	    }
	}
	if (chpow_flag == XTRUE)
	    tspgpow = pow(10.0, vec1->data[0]) * (frame * fs / 1000.0)
	        / (double)fftl;
	if (mel_flag == XTRUE) {
	    // mel cepstrum -> cepstrum
	    //vec2 = xcep2mcep(vec1, fftl / 2, fftl, XTRUE, XTRUE);
      vec2 = xcep2mpmcep(vec1, fftl / 2, fftl, XTRUE, XTRUE, alpha);
	    // cepstrum -> spectrum
	    vec = xget_cep2vec(vec2, fftl);
	    // memory free
	    xdvfree(vec2);
	} else {
	    // cepstrum -> spectrum
	    vec = xget_cep2vec(vec1, fftl);
	}
	if (chpow_flag == XTRUE) {
	    for (l = 0, spgpow = 0.0; l < vec->length; l++)
	        spgpow += vec->data[l] * vec->data[l];
	    spgpow /= (double)vec->length;
	    if (spgpow > 0.0)
	        for (l = 0; l < vec->length; l++)
		    vec->data[l] *= sqrt(tspgpow / spgpow);
	}
	dmcopyrow(spg, k, vec);
	// memory free
	xdvfree(vec);
    }

    // close file
    fclose(fp);

    // memory free
    if (float_flag == XTRUE) xfvfree(tmpf);
    xdvfree(vec1);

    return spg;
}

