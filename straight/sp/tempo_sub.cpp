/*
 *      tempo_sub.c : tempo sub routine
 *	V30k18 (matlab)
 *
 *      	coded by T. Toda		2001/2/6
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "defs.h"
#include "basic.h"
#include "vector.h"
#include "voperate.h"
#include "filter.h"
#include "fileio.h"
#include "fft.h"
#include "memory.h"

#include "straight_sub.h"
#include "tempo_sub.h"

/* cut low noise (f < f0floor) */
void cleaninglownoise(DVECTOR x,
		      double fs,
		      double f0floor)
{
    long nn, flp, k;
    double flm;
    DVECTOR wlp = NODATA;
    DVECTOR tx = NODATA;
    DVECTOR ttx = NODATA;

    flm = 50.0;
    flp = (long)round(fs * flm / 1000.0);
    nn = x->length;
    wlp = fir1(flp * 2, f0floor / (fs / 2.0));

    wlp->data[flp] = wlp->data[flp] - 1.0;
    dvscoper(wlp, "*", -1.0);
    tx = xdvzeros(x->length + 2 * wlp->length);
    dvcopy(tx, x);
    ttx = xdvfftfiltm(wlp, tx, wlp->length * 2);
    for (k = 0; k < nn; k++) x->data[k] = ttx->data[k + flp];

    /* memory free */
    xdvfree(wlp);
    xdvfree(tx);
    xdvfree(ttx);

    return;
}

/* lowpass FIR digital filter by using Hamming window */
DVECTOR fir1(long len,		/* return [length + 1] */
	     double cutoff)	/* cut-off frequency */
{
    long k, length;
    double half;
    double value, a;
    DVECTOR filter;

    /* 0 <= cufoff <= 1 */
    cutoff = MIN(cutoff, 1.0);
    cutoff = MAX(cutoff, 0.0);

    half = (double)len / 2.0;
    length = len + 1;

    /* memory allocate */
    filter = xdvalloc(length);

    /* hamming window */
    for (k = 0; k < length; k++) {
	filter->data[k] = 0.54 - 0.46 * cos(2.0 * PI * (double)k / (double)(length - 1));
    }

    /* calculate lowpass filter */
    for (k = 0; k < length; k++) {
	a = PI * ((double)k - half);
	if (a == 0.0) {
	    value = cutoff;
	} else {
	    value = sin(PI * cutoff * ((double)k - half)) / a;
	}
	filter->data[k] *= value;
    }

    return filter;
}

/* decrease the sampling rate for a sequence (decimation) */
DVECTOR decimate(DVECTOR x,
		 long dn)
{
    long k;
    DVECTOR tx;
    DVECTOR ttx;
    DVECTOR dsx;
    DVECTOR firfilter;
    DVECTORS filter;

    /* memory allocation */
    dsx = xdvalloc((long)ceil((double)x->length / (double)dn));
    /* decimation */
    if (1) {    /* use chebyshev filter */
	/* 8-th order lowpass chebyshev filter */
	if ((filter = cheby(8, 0.05, 0.8 / (double)dn)) == NODATA) {
	    printmsg(stderr, "Failed: Chebyshev Filter\n");
	    return NODATA;
	}

	/* filtering */
	tx = xdvfiltfilt(filter->vector[0], filter->vector[1], x);

	/* down-sampling */
	for (k = 0; k < dsx->length; k++) dsx->data[k] = tx->data[k * dn];

	/* memory free */
	xdvfree(tx);
	xdvsfree(filter);
    } else {	/* use FIR filter */
	/* 30-th FIR lowpass filter */
	firfilter = fir1(30, 1.0 / (double)dn);

	/* filtering */
	tx = xdvzeros(x->length + 2 * firfilter->length);
	dvcopy(tx, x);
	ttx = xdvfftfilt(firfilter, tx, firfilter->length * 2);

	/* down-sampling */
	for (k = 0; k < dsx->length; k++)
	    dsx->data[k] = ttx->data[k * dn + 15];

	/* memory free */
	xdvfree(tx);
	xdvfree(ttx);
	xdvfree(firfilter);
    }

    return dsx;
}

/* Chebyshev digital and analog filter design */
DVECTORS cheby(long n,		/* order */
	       double rp,	/* passband ripple [dB] */
	       double cutoff)	/* cut-off frequency */

{
    double gain = 0.0;
    DVECTOR pv;
    DVECTORS filter;

    /* 0 <= cufoff <= 1 */
    cutoff = MIN(cutoff, 1.0);
    cutoff = MAX(cutoff, 0.0);

    /* 2 <= n <= 16 */
    if (n < 2 || n > 16) {
	printmsg(stderr, "2 <= Order <= 16\n");
	return NODATA;
    }

    /* calculate gain and pole of chebyshev filter */
    pv = chebap(n, rp, &gain);

    /* lowpass filter */
    gain *= pow(cutoff, (double)n);
    dvscoper(pv, "*", cutoff);
    
    /* convert analog into digital through bilinear transformation */
    filter = a2dbil(pv, gain, cutoff, XFALSE);	/* lowpass filter */

    /* memory free */
    xdvfree(pv);

    return filter;
}

/* Chebyshev analog lowpass filter prototype */
DVECTOR chebap(long n,		/* order */
	       double rp,	/* passband ripple [dB] */
	       double *gain)	/* gain */
{
    long l, l2;
    double e, e2, alpha, alpha1, alpha2, a, b, w;
    DVECTOR pv;
    
    /* calculate gain and pole of chebyshev filter */
    e2 = pow(10.0, rp / 10.0) - 1.0;
    e = sqrt(e2);
    alpha = 1.0 / e + sqrt(1.0 + 1.0 / e2);
    /* gain */
    *gain = 1.0 / e * pow(0.5, (double)(n - 1));
    /* pole */
    alpha1 = pow(alpha, 1.0 / (double)n);
    alpha2 = pow(alpha, -1.0 / (double)n);
    a = (alpha1 - alpha2) / 2.0;
    b = (alpha1 + alpha2) / 2.0;
    pv = xdvrialloc(n);
    for (l = 0; l < n / 2; l++) {
	w = ((double)n + 1.0 + 2.0 * (double)l) / (2.0 * (double)n) * PI;
	l2 = l * 2;
	pv->data[l2] = a * cos(w);
	pv->imag[l2] = b * sin(w);
	pv->data[l2 + 1] = pv->data[l2];
	pv->imag[l2 + 1] = -1.0 * pv->imag[l2];
    }
    if ((l2 = l * 2) < n) {
	w = ((double)n + 1.0 + 2.0 * (double)l) / (2.0 * (double)n) * PI;
	pv->data[l2] = a * cos(w);
	pv->imag[l2] = 0.0;
    }

    return pv;
}

/* Bilinear transformation for filter (analog -> digital) */
DVECTORS a2dbil(DVECTOR pv,	/* pole */
		double gain,	/* gain */
		double cutoff,	/* cutoff frequency */
		XBOOL hp_flag)	/* highpass filter */
{
    long n, polydnum, l, l2, rn, cn, k;
    double r2, rp, ip, mapcoef, lphpcoef;
    DVECTOR vec1d;
    DVECTOR vec2d;
    DVECTOR vec1n;
    DVECTOR vec2n;
    DMATRIX polyd;
    DMATRIX polyn;
    DMATRICES calmatd;
    DMATRICES calmatn;
    DVECTORS coef;

    /* order */
    n = pv->length;

    mapcoef = PI / 2.0;
    /* match frequency */
    if (cutoff != 0.0) mapcoef = tan(cutoff * mapcoef) / cutoff;

    /* gain */
    if (hp_flag == XFALSE) {	/* lowpass filter */
	gain *= pow(mapcoef, (double)n);
	lphpcoef = 1.0;
    } else {			/* highpass filter */
	lphpcoef = -1.0;
    }
    
    /* polynomial z^-2 */
    polydnum = (long)pow(2.0, (double)(nextpow2(n) - 1));
    polyd = xdmzeros(polydnum, 3);
    polyn = xdmzeros(polydnum, 3);
    for (l = 0; l < n / 2; l++) {	/* real + imag, real - imag */
	l2 = l * 2;
	r2 = pv->data[l2] * 2.0;
	rp = pv->data[l2] * pv->data[l2];
	ip = pv->imag[l2] * pv->imag[l2];
	/* z^0 */
	polyd->data[l][0] = 1.0 - mapcoef * r2 + pow(mapcoef, 2.0) * (rp + ip);
	polyn->data[l][0] = 1.0;
	/* z^-1 */
	polyd->data[l][1] = -2.0 + 2.0 * pow(mapcoef, 2.0) * (rp + ip);
	polyn->data[l][1] = lphpcoef * 2.0;
	/* z^-2 */
	polyd->data[l][2] = 1.0 + mapcoef * r2 + pow(mapcoef, 2.0) * (rp + ip);
	polyn->data[l][2] = 1.0;
    }
    if ((l2 = l * 2) < n) {	/* imag = 0 */
	/* z^0 */
	polyd->data[l][0] = 1.0 - mapcoef * pv->data[l2];
	polyn->data[l][0] = 1.0;
	/* z^-1 */
	polyd->data[l][1] = -1.0 - mapcoef * pv->data[l2];
	polyn->data[l][1] = lphpcoef;

	l++;
    }
    for (; l < polydnum; l++) {
	polyd->data[l][0] = 1.0;
	polyn->data[l][0] = 1.0;
    }

    for (k = 1; k < nextpow2(n); k++) {
	/* calculate matrices */
	calmatd = xdmsalloc(polyd->row / 2);
	calmatn = xdmsalloc(calmatd->num_matrix);
	for (l = 0; l < calmatd->num_matrix; l++) {
	    calmatd->matrix[l] = xdmalloc(polyd->col, polyd->col);
	    calmatn->matrix[l] = xdmalloc(polyn->col, polyn->col);
	    l2 = l * 2;
	    vec1d = xdmextractrow(polyd, l2);
	    vec2d = xdmextractrow(polyd, l2 + 1);
	    vec1n = xdmextractrow(polyn, l2);
	    vec2n = xdmextractrow(polyn, l2 + 1);
	    for (rn = 0; rn < polyd->col; rn++) {
		for (cn = 0; cn < polyd->col; cn++) {
		    calmatd->matrix[l]->data[rn][cn]
			= vec1d->data[rn] * vec2d->data[cn];
		    calmatn->matrix[l]->data[rn][cn]
			= vec1n->data[rn] * vec2n->data[cn];
		}
	    }
	    xdvfree(vec1d);
	    xdvfree(vec2d);
	    xdvfree(vec1n);
	    xdvfree(vec2n);
	}
	xdmfree(polyd);
	xdmfree(polyn);

	/* calculate polynomial */
	polyd = xdmzeros(l, (long)pow(2.0, (double)(k + 1)) + 1);
	polyn = xdmzeros(l, polyd->col);
	for (l = 0; l < polyd->row; l++) {
	    for (rn = 0; rn < calmatd->matrix[l]->row; rn++) {
		for (cn = 0; cn < calmatd->matrix[l]->col; cn++) {
		    polyd->data[l][rn + cn] += calmatd->matrix[l]->data[rn][cn];
		    polyn->data[l][rn + cn] += calmatn->matrix[l]->data[rn][cn];
		}
	    }
	}
	xdmsfree(calmatd);
	xdmsfree(calmatn);
    }

    /* Filter-coefficient */
    coef = xdvsalloc(2);
    coef->vector[0] = xdvalloc(n + 1);
    coef->vector[1] = xdvalloc(n + 1);
    for (l = 0; l < n + 1; l++) {
	coef->vector[0]->data[l]
	    = polyn->data[0][l] * gain / polyd->data[0][0];
	coef->vector[1]->data[l] = polyd->data[0][l] / polyd->data[0][0];
    }

    /* memory free */
    xdmfree(polyd);
    xdmfree(polyn);

    return coef;
}

/*  Zero-phase forward and reverse digital filtering */
DVECTOR xdvfiltfilt(DVECTOR b,	/* filter-coefficient (numerator) */
		    DVECTOR a,	/* filter-coefficient (denominator) */
		    DVECTOR x)	/* input data */
{
    long l, k;
    double px1, px2, py1, py2;
    DVECTOR tmp;
    DVECTOR pxv;
    DVECTOR pyv;
    DVECTOR fyv;
    DVECTOR byv;

    /* normalize filter-coefficient */
    if (a->data[0] != 1.0) {
	printmsg(stderr, "normalize filter-coefficient\n");
	dvscoper(b, "/", a->data[0]);
	dvscoper(a, "/", a->data[0]);
	if (a->length > b->length) {
	    tmp = xdvzeros(a->length);
	    dvcopy(tmp, b);
	    xdvfree(b);
	    b = xdvclone(tmp);
	    xdvfree(tmp);
	} else if (b->length > a->length) {
	    tmp = xdvzeros(b->length);
	    dvcopy(tmp, a);
	    xdvfree(a);
	    a = xdvclone(tmp);
	    xdvfree(tmp);
	}
    }

    /* memory allocation */
    fyv = xdvalloc(x->length);
    byv = xdvalloc(x->length);
    pxv = xdvzeros(b->length - 1);
    pyv = xdvzeros(a->length - 1);

    /* forward filtering */
    for (l = 0; l < x->length; l++) {
	fyv->data[l] = b->data[0] * x->data[l];
	px1 = pxv->data[0];	py1 = pyv->data[0];
	px2 = px1;		py2 = py1;
	for (k = 0; k < pxv->length - 1; k++) {
	    fyv->data[l] += b->data[k + 1] * px1;
	    fyv->data[l] -= a->data[k + 1] * py1;
	    px1 = pxv->data[k + 1];	py1 = pyv->data[k + 1];
	    pxv->data[k + 1] = px2;	pyv->data[k + 1] = py2;
	    px2 = px1;			py2 = py1;
	}
	fyv->data[l] += b->data[k + 1] * px1;
	fyv->data[l] -= a->data[k + 1] * py1;
	pxv->data[0] = x->data[l];	pyv->data[0] = fyv->data[l];
    }
    for (l = 0; l < pxv->length; l++) {
	pxv->data[l] = 0.0;
	pyv->data[l] = 0.0;
    }
    /* backward filtering */
    for (l = x->length - 1; l >= 0; l--) {
	byv->data[l] = b->data[0] * fyv->data[l];
	px1 = pxv->data[0];	py1 = pyv->data[0];
	px2 = px1;		py2 = py1;
	for (k = 0; k < pxv->length - 1; k++) {
	    byv->data[l] += b->data[k + 1] * px1;
	    byv->data[l] -= a->data[k + 1] * py1;
	    px1 = pxv->data[k + 1];	py1 = pyv->data[k + 1];
	    pxv->data[k + 1] = px2;	pyv->data[k + 1] = py2;
	    px2 = px1;			py2 = py1;
	}
	byv->data[l] += b->data[k + 1] * px1;
	byv->data[l] -= a->data[k + 1] * py1;
	pxv->data[0] = fyv->data[l];	pyv->data[0] = byv->data[l];
    }

    /* memory free */
    xdvfree(pxv);
    xdvfree(pyv);
    xdvfree(fyv);

    return byv;
}

/* Dual waveleta analysis using cardinal spline manipulation */
DMATRIX multanalytFineCSPB(DVECTOR x,		/* input signal */
			   double fs,		/* sampling frequency */
			   double f0floor,	/* lower bound for pitch */
			   long nvc,	/* number of total voices */
			   long nvo,	/* number of voices in an octave */
			   double mu,	/* temporal stretch factor */
			   int mlt)	/* farmonic ID# */
{
    long l, nx, wl, ii, wbias;
    double t0, lmx, mpv;
    DVECTOR gent;
    DVECTOR tx;
    DVECTOR t;
    DVECTOR wwd;
    DVECTOR pmtmp;
    DMATRIX pm;

    t0 = 1.0 / f0floor;
    lmx = round(6.0 * t0 * fs * mu);
    wl = POW2(ceil(log(lmx) / log(2.0)));
    nx = x->length;
    tx = xdvzeros(x->length + wl);
    dvcopy(tx, x);
    gent = xdvalloc(wl);
    for (l = 0; l < gent->length; l++)
	gent->data[l] = ((double)l + 1.0 - wl / 2.0) / fs;
    /* memory allocation */
    pm = xdmrizeros(nvc, nx);
    mpv = 1.0;

    for (ii = 0; ii < nvc; ii++) {
	/* wavelet */
	t = cspb_xgetiregion(gent, t0, mpv, mu);
	wwd = cspb_xgetwavelet(t, t0, mu, mlt);

	wbias = (long)round((double)(wwd->length - 1) / 2.0);

	/* filtering */
	pmtmp = xdvfftfiltm(wwd, tx, wwd->length * 2);

	dvscoper(pmtmp, "*", sqrt(mpv));
	dmpasterow(pm, ii, pmtmp, -wbias, nx + wbias, 0);

	mpv = mpv * pow(2.0, 1.0 / (double)nvo);

	/* memory free */
	xdvfree(t);
	xdvfree(wwd);
	xdvfree(pmtmp);
    }
    /* memory free */
    xdvfree(gent);
    xdvfree(tx);

    return pm;
}

DVECTOR cspb_xgetiregion(DVECTOR gent,
			 double t0,
			 double mpv,
			 double mu)
{
    long k;
    LVECTOR tbi;
    DVECTOR tb;
    DVECTOR tb2;
    DVECTOR t;

    tb = xdvalloc(gent->length);
    tb2 = xdvalloc(gent->length);

    for (k = 0; k < tb->length; k++) {
	tb->data[k] = gent->data[k] * mpv;
	tb2->data[k] = fabs(tb->data[k]);
    }

    tbi = xdvscfind(tb2, "<", 3.5 * mu * t0);
    t = xdvremap(tb, tbi);

    /* memory free */
    xlvfree(tbi);
    xdvfree(tb);
    xdvfree(tb2);

    return t;
}

DVECTOR cspb_xgetwavelet(DVECTOR t,
			 double t0,
			 double mu,
			 int mlt)
{
    long l, len, tidx;
    double wbias, tmp;
    LVECTOR wwdbi;
    DVECTOR wd1;
    DVECTOR wd2;
    DVECTOR wdb2;
    DVECTOR wwdb;
    DVECTOR wwd;
    
    wd1 = xdvalloc(t->length);
    wdb2 = xdvalloc(t->length);

    for (l = 0, len = 0; l < wd1->length; l++) {
	tmp = t->data[l] / t0 / mu;
	/* gaussian window */
	wd1->data[l] = exp(-PI * pow(tmp, 2.0));
	/* triangular window */
	wdb2->data[l] = MAX(0.0, 1.0 - fabs(tmp));
	if (wdb2->data[l] != 0.0) len++;
    }
    /* shift triangular window */
    wd2 = xdvalloc(len);
    for (l = 0, len = 0; l < wdb2->length && len < wd2->length; l++) {
	if (wdb2->data[l] != 0.0) {
	    wd2->data[len] = wdb2->data[l];
	    len++;
	}
    }
    /* convolution */
    wwdb = xdvconv(wd2, wd1);
    dvabs(wwdb);
    wwdbi = xdvscfind(wwdb, ">", 0.00001);
    wwd = xdvremap(wwdb, wwdbi);

    wbias = round((double)(wwd->length - 1) / 2.0);
    dvialloc(wwd);

    /* wavelet */
    for (l = 0; l < wwd->length; l++) {
	tidx = (long)round((double)l - wbias + (double)t->length / 2.0);
	if (tidx >= 0 && tidx < t->length) {
	    tmp = 2.0 * PI * (double)mlt * t->data[tidx] / t0;
	} else {
	    tmp = 0.0;
	}
	wwd->imag[l] = wwd->data[l] * sin(tmp);
	wwd->data[l] = wwd->data[l] * cos(tmp);
    }

    /* memory free */
    xlvfree(wwdbi);
    xdvfree(wd1);
    xdvfree(wd2);
    xdvfree(wdb2);
    xdvfree(wwdb);

    return wwd;
}

/* Wavelet to instantaneous frequency map */
DMATRIX zwvlt2ifq(DMATRIX pm,
		  double fs)
{
    long r, c;
    DVECTOR vec;
    DVECTOR amp;
    DVECTOR pvec;
    DVECTOR pamp;
    DVECTOR pifv;
    DMATRIX pif;

    /* memory allocation */
    pif = xdmalloc(pm->row, pm->col);

    pvec = xdmextractcol(pm, 0);
    pamp = xdvabs(pvec);
    dvoper(pvec, "/", pamp);
    for (c = 1; c < pm->col; c++) {
	vec = xdmextractcol(pm, c);
	amp = xdvabs(vec);
	dvoper(vec, "/", amp);

	pifv = xdvoper(vec, "-", pvec);
	dvabs(pifv);

	for (r = 0; r < pifv->length; r++)
	    pif->data[r][c] = fs / PI * asin(pifv->data[r] / 2.0);

	dvcopy(pvec, vec);

	/* memory free */
	xdvfree(vec);
	xdvfree(amp);
	xdvfree(pifv);
    }
    
    for (r = 0; r < pif->row; r++) pif->data[r][0] = pif->data[r][1];

    /* memory free */
    xdvfree(pvec);
    xdvfree(pamp);

    return pif;
}

/* Instantaneous frequency 2 geometric parameters */
DMATRICES zifq2gpm2(DMATRIX pif,
		    double f0floor,
		    long nvo)
{
    long k, l;
    double c;
    DVECTOR fx;
    DMATRIX g;
    DMATRIX h;
    DMATRICES slppbl;

    /* memory allocation */
    fx = xdvalloc(pif->row);
    for (k = 0; k < fx->length; k++)
	fx->data[k] = f0floor * pow(2.0, (double)k / (double)nvo) * 2.0 * PI;

    /* memory allocation */
    g = xdmalloc(3, 3);
    c = pow(2.0, 1.0 / (double)nvo);
    g->data[0][0] = 1.0 / c / c; g->data[0][1] = 1.0 / c; g->data[0][2] = 1.0;
    g->data[1][0] = 1.0;	 g->data[1][1] = 1.0;	  g->data[1][2] = 1.0;
    g->data[2][0] = c * c;	 g->data[2][1] = c;	  g->data[2][2] = 1.0;
    h = ss_xinvmat_svd(g, 1.0e-12);

    /* memory allocation */
    slppbl = xdmsalloc(2);
    slppbl->matrix[0] = xdmalloc(pif->row, pif->col);
    for (k = 0; k < slppbl->matrix[0]->row - 2; k++) {
	for (l = 0; l < slppbl->matrix[0]->col; l++) {
	    slppbl->matrix[0]->data[k + 1][l] = ((pif->data[k + 1][l] - pif->data[k][l]) / (1.0 - 1.0 / c) + (pif->data[k + 2][l] - pif->data[k + 1][l]) / (c - 1.0)) / 2.0;
	}
    }
    for (l = 0; l < slppbl->matrix[0]->col; l++) {
	slppbl->matrix[0]->data[0][l] = slppbl->matrix[0]->data[1][l];
	slppbl->matrix[0]->data[slppbl->matrix[0]->row - 1][l] = slppbl->matrix[0]->data[slppbl->matrix[0]->row - 2][l];
    }

    /* memory allocation */
    slppbl->matrix[1] = xdmalloc(pif->row, pif->col);
    for (k = 0; k < slppbl->matrix[1]->row - 2; k++) {
	for (l = 0; l < slppbl->matrix[1]->col; l++) {
	    slppbl->matrix[1]->data[k + 1][l] = pif->data[k][l] * h->data[1][0] + pif->data[k + 1][l] * h->data[1][1] + pif->data[k + 2][l] * h->data[1][2];
	}
    }

    for (l = 0; l < slppbl->matrix[1]->col; l++) {
	slppbl->matrix[1]->data[0][l] = slppbl->matrix[1]->data[1][l];
	slppbl->matrix[1]->data[slppbl->matrix[1]->row - 1][l] = slppbl->matrix[1]->data[slppbl->matrix[1]->row - 2][l];
    }

    for (k = 0; k < pif->row; k++) {
	for (l = 0; l < pif->col; l++) {
	    slppbl->matrix[0]->data[k][l] /= fx->data[k];
	    slppbl->matrix[1]->data[k][l] /= fx->data[k];
	}
    }

    /* memory free */
    xdvfree(fx);
    xdmfree(g);
    xdmfree(h);

    return slppbl;
}

double *znrmlcf2(double f)
{
    long k;
    double n = 100.0;
    double *c;
    DVECTOR x;
    DVECTOR xx;
    DVECTOR tmp;
    DVECTOR g;
    DVECTOR dgs;

    c = xalloc(2, double);
    x = xdvalloc((long)n * 3 + 1);
    for (k = 0; k < x->length; k++)
	x->data[k] = (double)k / n;

    g = zGcBs(x, 0.0);

    dgs = xdvalloc(g->length);
    for (k = 0; k < g->length - 1; k++)
	dgs->data[k] = (g->data[k + 1] - g->data[k]) * n / 2.0 / PI / f;
    dgs->data[g->length - 1] = 0.0;

    xx = xdvscoper(x, "*", 2.0 * PI * f);
    tmp = xdvoper(xx, "*", dgs);
    dvscoper(tmp, "^", 2.0);
    c[0] = dvsum(tmp) / n * 2.0;
    dvscoper(xx, "^", 2.0);
    dvoper(xx, "*", dgs);
    dvscoper(xx, "^", 2.0);
    c[1] = dvsum(xx) / n * 2.0;

    /* memory free */
    xdvfree(x);
    xdvfree(xx);
    xdvfree(tmp);
    xdvfree(g);
    xdvfree(dgs);

    return c;
}

DVECTOR zGcBs(DVECTOR x, double k)
{
    long l;
    DVECTOR tt;
    DVECTOR p;

    tt = xdvclone(x);
    dvscoper(tt, "+", 0.0000001);

    p = xdvalloc(x->length);
    for (l = 0; l < p->length; l++) {
	p->data[l] = pow(tt->data[l], k) * exp(-PI * pow(tt->data[l], 2.0)) * pow(sin(PI * tt->data[l] + 0.0001) / (PI * tt->data[l] + 0.0001), 2.0);
    }

    /* memory free */
    xdvfree(tt);
    
    return p;
}

DMATRIX zsmoothmapB(DMATRIX map,
		    double fs,
		    double f0floor,
		    long nvo,
		    double mu,
		    double mlim,
		    double pex)
{
    long l, ii, wbias;
    double t0, lmx, mpv = 1.0;
    double sumwd1, sumwd2, wl;
    LVECTOR tbi;
    LVECTOR iiv;
    DVECTOR gent;
    DVECTOR t;
    DVECTOR tb;
    DVECTOR wd1;
    DVECTOR wd2;
    DVECTOR tm;
    DVECTOR tmb;
    DVECTOR mapvec;
    DVECTOR mapzt;
    DVECTOR tmzt;
    DMATRIX smap;

    t0 = 1.0 / f0floor;
    lmx = round(6.0 * t0 * fs * mu);
    wl = POW2(ceil(log(lmx) / log(2.0)));
    gent = xdvalloc((long)wl);
    for (l = 0; l < gent->length; l++)
	gent->data[l] = ((double)l + 1.0 - wl / 2.0) / fs;

    smap = ss_xdmclone(map);
    iiv = xlvalloc(map->col);
    for (l = 0; l < iiv->length; l++) iiv->data[l] = l;

    for (ii = 0; ii < map->row; ii++) {
	/* gaussian window */
	tb = xdvscoper(gent, "*", mpv);
	dvabs(tb);
	tbi = xdvscfind(tb, "<", 3.5 * mu * t0);
	t = xdvremap(tb, tbi);
	wbias = (long)round((double)(t->length - 1) /2.0);
	wd1 = xdvalloc(t->length);	wd2 = xdvalloc(t->length);
	for (l = 0, sumwd1 = 0.0, sumwd2 = 0.0; l < wd1->length; l++) {
	    wd1->data[l]
		= exp(-PI * pow(t->data[l] / (t0 * (1.0 - pex)) / mu, 2.0));
	    wd2->data[l]
		= exp(-PI * pow(t->data[l] / (t0 * (1.0 + pex)) / mu, 2.0));
	    sumwd1 += wd1->data[l];
	    sumwd2 += wd2->data[l];
	}
	dvscoper(wd1, "/", sumwd1);	dvscoper(wd2, "/", sumwd2);

	/* filtering */
	mapvec = xdmextractrow(map, ii);
	mapzt = xdvzeros(mapvec->length + gent->length);
	dvcopy(mapzt, mapvec);
	tmb = xdvfftfiltm(wd1, mapzt, wd1->length * 2);
	tmzt = xdvzeros(iiv->length + gent->length);
	for (l = 0; l < iiv->length; l++) {
	    /* floating error */
	    if (tmb->data[iiv->data[l] + wbias] < 0.000001)
		tmb->data[iiv->data[l] + wbias] = 0.000001;
	    tmzt->data[l] = 1.0 / tmb->data[iiv->data[l] + wbias];
	}
	tm = xdvfftfiltm(wd2, tmzt, wd2->length * 2);
	for (l = 0; l < iiv->length; l++)
	    smap->data[ii][l] = 1.0 / tm->data[iiv->data[l] + wbias];

	if (t0 * mu / mpv * 1000.0 > mlim)
	    mpv *= pow(2.0, (1.0 / (double)nvo));

	/* memory free */
	xlvfree(tbi);
	xdvfree(tb);
	xdvfree(t);
	xdvfree(wd1);
	xdvfree(wd2);
	xdvfree(mapvec);
	xdvfree(mapzt);
	xdvfree(tmb);
	xdvfree(tmzt);
	xdvfree(tm);
    }

    /* memory free */
    xlvfree(iiv);
    xdvfree(gent);

    return smap;
}

DVECTORS zfixpfreq3(DVECTOR fxx,
		    DVECTOR pif2,
		    DVECTOR mmp,
		    DVECTOR dfv,
		    DVECTOR pm,
		    XBOOL allparam_flag)	/* XFALSE */
{
    long k, nn, idx;
    LVECTOR iix = NODATA;
    LVECTOR ixx = NODATA;
    LVECTOR fpi = NODATA;
    DVECTOR fp = NODATA;
    DVECTOR aav = NODATA;
    DVECTOR cd1 = NODATA;
    DVECTOR cd2 = NODATA;
    DVECTOR cdd1 = NODATA;
    DVECTORS f0info = NODATA;

    if (allparam_flag != XFALSE) aav = xdvabs(pm);
    nn = fxx->length;

    /* memory allocation */
    iix = xlvalloc(nn);
    cd1 = xdvalloc(nn);
    cd2 = xdvalloc(nn);
    cdd1 = xdvalloc(nn);
    fp = xdvalloc(nn);
    for (k = 0; k < nn; k++) {
	iix->data[k] = k;
	cd1->data[k] = pif2->data[k] - fxx->data[k];
    }
    for (k = 0; k < nn - 1; k++) {
	cd2->data[k] = cd1->data[k + 1] - cd1->data[k];
	cdd1->data[k] = cd1->data[k + 1];
    }
    cd2->data[nn - 1] = cd1->data[nn - 1] - cd1->data[nn - 2];
    cdd1->data[nn - 1] = cd1->data[nn - 1];
    
    for (k = 0; k < nn; k++) {
	fp->data[k] = (double)((cd1->data[k] * cdd1->data[k] < 0.0) * (cd2->data[k] < 0.0));
    }
    fpi = xdvscfind(fp, ">", 0.0);
    ixx = xlvremap(iix, fpi);
    
    /* memory allocation */
    if (allparam_flag == XFALSE) {
	f0info = xdvsalloc(2);
    } else {
	f0info = xdvsalloc(4);
	f0info->vector[2] = xdvalloc(ixx->length);
	f0info->vector[3] = xdvalloc(ixx->length);
    }
    f0info->vector[0] = xdvalloc(ixx->length);
    f0info->vector[1] = xdvalloc(ixx->length);
    for (k = 0; k < ixx->length; k++) {
	idx = ixx->data[k];
	/* fixed point frequency vector */
	f0info->vector[0]->data[k] = pif2->data[idx] + (pif2->data[idx + 1] - pif2->data[idx]) * cd1->data[idx] / (cd1->data[idx] - cdd1->data[idx]);
	/* relative interfering energy vector */
	f0info->vector[1]->data[k] = mmp->data[idx] + (mmp->data[idx + 1] - mmp->data[idx]) * (f0info->vector[0]->data[k] - fxx->data[idx]) / (fxx->data[idx + 1] - fxx->data[idx]);
	if (allparam_flag != XFALSE) {
	    /* fixed point slope vector */
	    f0info->vector[2]->data[k] = dfv->data[idx] + (dfv->data[idx + 1] - dfv->data[idx]) * (f0info->vector[0]->data[k] - fxx->data[idx]) / (fxx->data[idx + 1] - fxx->data[idx]);
	    /* amplitude list for fixed points */
	    f0info->vector[3]->data[k] = aav->data[idx] + (aav->data[idx + 1] - aav->data[idx]) * (f0info->vector[0]->data[k] - fxx->data[idx]) / (fxx->data[idx + 1] - fxx->data[idx]);
	}
    }

    /* memory free */
    xlvfree(iix);
    xlvfree(ixx);
    xlvfree(fpi);
    xdvfree(fp);
    xdvfree(cd1);
    xdvfree(cd2);
    xdvfree(cdd1);
    if (allparam_flag != XFALSE) xdvfree(aav);

    return f0info;
}

DVECTORS plotcpower(DVECTOR x,
		    double fs,
		    double shiftm)
{
    long k, fl, nn, flp, len, idx;
    double flm = 8.0;	/* temporal resolution in ms */
    double flpm, sum, didx;
    LVECTOR idxv;
    DVECTOR w;
    DVECTOR wlp;
    DVECTOR tx;
    DVECTOR txp;
    DVECTOR txb;
    DVECTOR ttx;
    DVECTOR pw;
    DVECTOR b;
    DVECTOR xh;
    DVECTOR pwh;
    DVECTORS pwinfo;

    fl = (long)round(flm * fs / 1000.0);
    pwinfo = xdvsalloc(2);
    
    /* Hanning window */
    w = xdvalloc(2 * fl + 1);
    for (k = 0, sum = 0.0; k < w->length; k++) {
	w->data[k] = 0.5 - 0.5 * cos(2.0 * (double)(k + 1)* PI / (double)(w->length + 1));
	sum += w->data[k];
    }
    dvscoper(w, "/", sum);
    nn = x->length;

    flpm = 40.0;
    flp = (long)round(flpm * fs / 1000.0);
    /* lowpass filter -> highpass filter */
    wlp = fir1(flp * 2, 70.0 / (fs / 2.0));
    wlp->data[flp+1] =  wlp->data[flp+1] - 1.0;
    dvscoper(wlp, "*", -1.0);

    /* cut low noise */
    txb = xdvzeros(nn + 2 * wlp->length);
    dvcopy(txb, x);
    /* filtering (highpass filter 70 [Hz] < freq) */
    ttx = xdvfftfiltm(wlp, txb, wlp->length * 2);
    txp = xdvzeros(nn + 2 * w->length);
    tx = xdvzeros(nn + 2 * w->length);
    for (k = 0; k < nn; k++) {
	tx->data[k] = ttx->data[k + flp];
	txp->data[k] = pow(ttx->data[k + flp], 2.0);
    }
    /* memory free */
    xdvfree(wlp);
    xdvfree(ttx);
    xdvfree(txb);

    /* filtering (Hanning window) */
    pw = xdvfftfiltm(w, txp, w->length * 2);
    idxv = xlvalloc(nn);
    for (didx = 0.0, idxv->data[0] = 0, len = 1;
	 didx < (double)(nn - 1); didx += shiftm * fs / 1000.0) {
	idx = (long)round(didx);
	if (idxv->data[len - 1] < idx) {
	    idxv->data[len] = idx;
	    len++;
	}
    }
    /* total power */
    pwinfo->vector[0] = xdvalloc(len);
    for (k = 0; k < len; k++)
	pwinfo->vector[0]->data[k] = pw->data[idxv->data[k] + fl] + 0.0001;
    /* memory free */
    xdvfree(txp);
    xdvfree(pw);

    /* bandpass-filter */
    b = fir1bp(2 * fl + 1, 0.0001, 3000.0/ (fs /2.0));
    b->data[fl] -= 1.0;
    /* filtering (band-cut filter 3000 [Hz] < freq) */
    xh = xdvfftfiltm(b, tx, b->length * 2);
    txp = xdvzeros(nn + 10 * w->length);
    for (k = 0; k < nn; k++) txp->data[k] = pow(xh->data[k + fl], 2.0);

    /* filtering (Hanning window) */
    pwh = xdvfftfilt(w, txp, w->length * 2);
    /* power in higher frequency range */
    pwinfo->vector[1] = xdvalloc(len);
    for (k = 0; k < len; k++)
	pwinfo->vector[1]->data[k] = pwh->data[idxv->data[k] + fl];
    /* memory free */
    xdvfree(w);
    xdvfree(b);
    xdvfree(tx);
    xdvfree(txp);
    xdvfree(xh);
    xdvfree(pwh);
    xlvfree(idxv);

    return pwinfo;
}

/* Bandpass FIR Filter (start-endf) */
DVECTOR fir1bp(long len,
	       double startf,
	       double endf)
{
    long k;
    double half, cutoff;
    double fmf, sum;
    DVECTOR filter;

    if (startf * endf < 0.0 || startf * endf > 1.0) {
	printmsg(stderr, "0 <= freq <= 1, freq = %f, %f\n", startf, endf);
	exit(1);
    } else if (startf > endf) {
	printmsg(stderr, "start-freq(%f) < end-freq(%f)\n", startf, endf);
	exit(1);
    }

    cutoff = endf - startf;
    cutoff /= 2.0;
    /* lowpass filter */
    filter = fir1(len, cutoff);
    /* FM */
    fmf = startf + cutoff;
    half = (double)len / 2.0;
    for (k = 0, sum = 0.0; k < filter->length; k++) {
	filter->data[k] *=
	    sin(PI * fmf * ((double)k - half + 1.0 / 2.0 / fmf));
	sum += filter->data[k];
    }
    dvscoper(filter, "/", sum);

    return filter;
}

/* F0 trajectory tracker */
DVECTORS f0track5(DMATRICES f0infomat,	/* f0v, vrv, (dfv, aav) */
		  DVECTORS pwinfo,	/* pwt, pwh */
		  double shiftm,
		  XBOOL allparam_flag)	/* XFALSE */
{
    int von;
    long rn, cn, nn, mm, k, ii, jj, jxx = 0;
    double hth, lth, bklm, lalm, bkls, lals, thf0j, f0ref, gomi;
    LVECTOR ixx;
    DVECTOR mxvr;
    DVECTOR vrvvec;
    DVECTOR f0vvec;
    DVECTOR htr;
    DMATRIX vrv;
    DVECTORS f0trinfo;

    vrv = xdmalloc(f0infomat->matrix[1]->row, f0infomat->matrix[1]->col);
    for (rn = 0; rn < vrv->row; rn++) {
	for (cn = 0; cn < vrv->col; cn++)
	    vrv->data[rn][cn] = sqrt(f0infomat->matrix[1]->data[rn][cn]);
    }
    nn = vrv->row;	mm = vrv->col;
    mm = MIN(mm, pwinfo->vector[0]->length);

    /* memory allocation */
    if (allparam_flag == XFALSE) {
	f0trinfo = xdvsalloc(2);
    } else {
	f0trinfo = xdvsalloc(4);
	f0trinfo->vector[2] = xdvones(mm);
	f0trinfo->vector[3] = xdvzeros(mm);
    }
    f0trinfo->vector[0] = xdvzeros(mm);
    f0trinfo->vector[1] = xdvones(mm);
 
    von = 0;
    mxvr = xdvalloc(vrv->col);
    ixx = xlvalloc(vrv->col);
    for (k = 0; k < vrv->col; k++) {
	vrvvec = xdmextractcol(vrv, k);
	mxvr->data[k] = dvmin(vrvvec, &ixx->data[k]);
	xdvfree(vrvvec);
    }
    hth = 0.12;	/* highly confident voiced threshould */
    lth = 0.9;	/* threshold to loose confidence */
    bklm = 100.0;	/*back track length for voicing decision */
    lalm = 10.0;	/* look ahead length for silence decision */
    bkls = bklm / shiftm;
    lals = lalm / shiftm;
    htr = xdvalloc(pwinfo->vector[0]->length);
    for (k = 0; k < htr->length; k++) {
	/* floating error */
	if (pwinfo->vector[1]->data[k] <= 0.0)
	    pwinfo->vector[1]->data[k] = 0.000001;
	if (pwinfo->vector[0]->data[k] <= 0.0)
	    pwinfo->vector[0]->data[k] = 0.000001;
	htr->data[k] = 10.0 * log10(pwinfo->vector[1]->data[k] / pwinfo->vector[0]->data[k]);
    }

    f0vvec = xdvalloc(f0infomat->matrix[0]->row);
    thf0j = 0.04 * sqrt(shiftm);	/* 4 % of F0 is the limit of jump */

    for (ii = 0, f0ref = 0.0; ii < mm; ii++) {
	if ((von == 0 && mxvr->data[ii] < hth) && htr->data[ii] < -3.0) {
	    von = 1;
	    /*start value for search */
	    f0ref = f0infomat->matrix[0]->data[ixx->data[ii]][ii];
	    for (jj = ii; jj >= 0 && (double)jj >= (double)ii - bkls; jj--) {
		for (rn = 0; rn < f0vvec->length; rn++)
		    f0vvec->data[rn] = fabs((f0infomat->matrix[0]->data[rn][jj] - f0ref) / f0ref);
		gomi = dvmin(f0vvec, &jxx);
		gomi += (f0ref > 10000.0) + (f0infomat->matrix[0]->data[jxx][jj] > 10000.0);
		if (((gomi > thf0j || vrv->data[jxx][jj] > lth) || htr->data[jj] > -3.0) * (f0infomat->matrix[0]->data[jxx][jj] < 1000.0)
		    && htr->data[jj] > -18.0) break;
		if (gomi > thf0j) break;
		f0trinfo->vector[0]->data[jj]
		    = f0infomat->matrix[0]->data[jxx][jj];
		f0trinfo->vector[1]->data[jj] = vrv->data[jxx][jj];
		if (allparam_flag != XFALSE) {
		    f0trinfo->vector[2]->data[jj]
			= f0infomat->matrix[2]->data[jxx][jj];
		    f0trinfo->vector[3]->data[jj]
			= f0infomat->matrix[3]->data[jxx][jj];
		}
		f0ref = f0trinfo->vector[0]->data[jj];
	    }
	    f0ref = f0infomat->matrix[0]->data[ixx->data[ii]][ii];
	}
	if (f0ref > 0.0 && f0ref < 10000.0) {
	    for (rn = 0; rn < f0vvec->length; rn++)
		f0vvec->data[rn] = fabs((f0infomat->matrix[0]->data[rn][ii] - f0ref) / f0ref);
	    gomi = dvmin(f0vvec, &jxx);
	} else {
	    gomi = 10.0;
	}

	if (von == 1 && (mxvr->data[ii] > hth)) {
	    for (jj = ii; jj < mm && (double)jj <= (double)ii + lals; jj++) {
		ii = jj;
		for (rn = 0; rn < f0vvec->length; rn++)
		    f0vvec->data[rn] = fabs((f0infomat->matrix[0]->data[rn][ii] - f0ref) / f0ref);
		gomi = dvmin(f0vvec, &jxx);
		gomi += (f0ref > 10000.0) + (f0infomat->matrix[0]->data[jxx][jj] > 10000.0);
		if (gomi < thf0j && (htr->data[ii] < -3.0) + (f0infomat->matrix[0]->data[jxx][ii] >= 1000.0)) {
		    f0trinfo->vector[0]->data[ii]
			= f0infomat->matrix[0]->data[jxx][ii];
		    f0trinfo->vector[1]->data[ii] = vrv->data[jxx][ii];
		    if (allparam_flag != XFALSE) {
			f0trinfo->vector[2]->data[ii]
			    = f0infomat->matrix[2]->data[jxx][ii];
			f0trinfo->vector[3]->data[ii]
			    = f0infomat->matrix[3]->data[jxx][ii];
		    }
		    f0ref = f0trinfo->vector[0]->data[ii];
		}
		if ((gomi > thf0j || vrv->data[jxx][ii] > lth) || (htr->data[ii] > -3.0) * (f0infomat->matrix[0]->data[jxx][ii] < 1000.0)) {
		    von = 0;
		    f0ref = 0.0;
		    break;
		}
	    }
	} else if (von == 1 && gomi < thf0j && (htr->data[ii] < -3.0) + (f0infomat->matrix[0]->data[jxx][ii] > 1000.0)) {
	    f0trinfo->vector[0]->data[ii]
		= f0infomat->matrix[0]->data[jxx][ii];
	    f0trinfo->vector[1]->data[ii] = vrv->data[jxx][ii];
	    if (allparam_flag != XFALSE) {
		f0trinfo->vector[2]->data[ii]
		    = f0infomat->matrix[2]->data[jxx][ii];
		f0trinfo->vector[3]->data[ii]
		    = f0infomat->matrix[3]->data[jxx][ii];
	    }
	    f0ref = f0trinfo->vector[0]->data[ii];
	} else {
	    von = 0;
	}
    }

    /* memory free */
    xlvfree(ixx);
    xdvfree(mxvr);
    xdvfree(f0vvec);
    xdvfree(htr);
    xdmfree(vrv);

    return f0trinfo;
}

/* F0 estimation refinement */
DVECTORS refineF02(DVECTOR x,	/* input waveform */
		   double fs,	/* sampling frequency [Hz] */
		   DVECTOR f0raw,	/* F0 candidate [Hz] */
		   long fftl,	/* FFT length */
		   double eta,	/* temporal stretch factor */
		   long nhmx,	/* highest harmonic number */
		   double shiftm,	/* frame shift period [ms] */
		   long nl,	/* lower frame number */
		   long nu,	/* upper frame number */
		   XBOOL allparam_flag)	/* XFALSE*/
{
    long k, nfr, lenx, len1, len2, nlo, kk;
    long wal, bias, ii, sml, smb, lidx;
    double xlinms, shiftl, f0t, tmp, *c, c1, c2;
    double sum, dfidx, iidx, vvv, fqv;
    LVECTOR w1i = NODATA;	LVECTOR wgi = NODATA;
    DVECTOR f0i = NODATA;	DVECTOR lx = NODATA;
    DVECTOR rr = NODATA;	DVECTOR w1 = NODATA;
    DVECTOR w1b = NODATA;	DVECTOR wg = NODATA;
    DVECTOR wgg = NODATA;	DVECTOR wo = NODATA;
    DVECTOR wa = NODATA;	DVECTOR xo = NODATA;
    DVECTOR xi = NODATA;	DVECTOR x0 = NODATA;
    DVECTOR x1 = NODATA;	DVECTOR x2 = NODATA;
    DVECTOR ff0 = NODATA;	DVECTOR ff1 = NODATA;
    DVECTOR ff2 = NODATA;	DVECTOR fd = NODATA;
    DVECTOR fd0 = NODATA;	DVECTOR crf = NODATA;
    DVECTOR crf0 = NODATA;	DVECTOR hann = NODATA;
    DVECTOR mmpvec = NODATA;	DVECTOR smmpbvec = NODATA;
    DVECTOR smmpvec = NODATA;	DVECTOR pwm1vec = NODATA;
    DVECTOR pwm2vec = NODATA;	DVECTOR spwmvec = NODATA;
    DVECTOR spfmvec = NODATA;	DVECTOR idx = NODATA;
    DVECTOR sum1v = NODATA;	DVECTOR sum2v = NODATA;
    DVECTOR sum3v = NODATA;
    DVECTORS reff0info = NODATA;
    DMATRIX pif = NODATA;	DMATRIX dpif = NODATA;
    DMATRIX pwm = NODATA;	DMATRIX slp = NODATA;
    DMATRIX dslp = NODATA;	DMATRIX mmp = NODATA;
    DMATRIX smmp = NODATA;	DMATRIX spif = NODATA;

    /* memory allocation */
    f0i = xdvalloc(f0raw->length);
    for (k = 0; k < f0raw->length; k++) {
	if (f0raw->data[k] == 0.0) {
	    f0i->data[k] = 160.0;
	} else {
	    f0i->data[k] = f0raw->data[k];
	}
    }

    xlinms = (double)x->length / fs * 1000.0;
    nfr = f0i->length;
    shiftl = shiftm / 1000.0 * fs;
    lenx = x->length;
    f0t = 100.0;

    /* memory allocation */
    lx = xdvzeros(fftl * 2 + lenx);
    dvpaste(lx, x, fftl, x->length, 0);
    rr = xdvrialloc(fftl);
    for (k = 0; k < fftl; k++) {
	tmp = (double)k / (double)fftl * 2.0 * PI;
	rr->data[k] = cos(tmp);
	rr->imag[k] = sin(tmp * -1.0);
    }

    /* make window */
    w1b = xdvalloc(fftl);
    w1i = xlvalloc(fftl);
    wg = xdvalloc(fftl);
    wgi = xlvalloc(fftl);
    for (k = 0, len1 = 0, len2 = 0; k < fftl; k++) {
	tmp = ((double)k + 1.0 - (double)fftl / 2.0) / fs * f0t / eta;
	/* triangular window */
	w1b->data[k] = MAX(0.0, 1.0 - fabs(tmp));
	/* gaissian window */
	wg->data[k] = exp(-PI * pow(tmp, 2.0));
	if (w1b->data[k] > 0.0) {
	    w1i->data[len1] = k;	len1++;
	}
	if (fabs(wg->data[k]) > 0.0002) {
	    wgi->data[len2] = k;	len2++;
	}
    }
    w1 = xdvalloc(len1 + len2);
    for (k = 0; k < len1; k++) w1->data[k] = w1b->data[w1i->data[k]];
    for (; k < w1->length; k++) w1->data[k] = 0.0;
    wgg = xdvalloc(len2);
    for (k = 0; k < len2; k++) wgg->data[k] = wg->data[wgi->data[k]];
    /* convolution */
    wo = xdvfftfiltm(wgg, w1, wgg->length * 2);

    /* memory free */
    xlvfree(w1i);    xlvfree(wgi);    xdvfree(w1);
    xdvfree(w1b);    xdvfree(wg);    xdvfree(wgg);

    /* memory allocation */
    xo = xdvalloc(wo->length);
    for (k = 0; k < xo->length; k++)
	xo->data[k] = (double)k / (double)(wo->length - 1);
    nlo = wo->length - 1;

    if (nl * nu < 0) {
	nl = 1;	nu = nfr;
    }

    /* memory allocation */
    pif = xdmzeros(fftl / 2 + 1, nfr);
    dpif = xdmzeros(pif->row, nfr);
    pwm = xdmzeros(pif->row, nfr);

    for (kk = nl - 1; kk < nu; kk++) {
	if (f0i->data[kk] < 40.0) f0i->data[kk] = 40.0;
	f0t = f0i->data[kk];

	/* memoroy allocation */
	for (tmp = 0.0, len1 = 0; tmp <= 1.0;
	     tmp += f0t / (double)nlo / 100.0) len1++;
	xi = xdvalloc(len1);
	for (k = 0; k < len1; k++)
	    xi->data[k] = (double)k * f0t / (double)nlo / 100.0;
	/* interpolation of window */
	if ((wa = interp1q(xo, wo, xi)) == NODATA) {
	    printmsg(stderr, "Error: refineF02\n");
	    return NODATA;
	}
	wal = wa->length;
	/* memory free */
	xdvfree(xi);

	bias = (long)round((double)fftl - (double)wal / 2.0 + (double)kk * shiftl);
	/* truncate the sequence */
	if (wal > fftl) wal = fftl;

	/* memory allocation */
	x0 = xdvalloc(wal); x1 = xdvalloc(wal); x2 = xdvalloc(wal);
	/* multiply window */
	for (k = 0; k < wal; k++) {
	    x0->data[k] = lx->data[k + bias - 1] * wa->data[k];
	    x1->data[k] = lx->data[k + bias] * wa->data[k];
	    x2->data[k] = lx->data[k + bias + 1] * wa->data[k];
	}
	/* FFT */
	ff0 = xdvfft(x0, fftl); ff1 = xdvfft(x1, fftl);	ff2 = xdvfft(x2, fftl);
	fd = xdvoper(ff2, "*", rr);	dvoper(fd, "-", ff1);
	fd0 = xdvoper(ff1, "*", rr);	dvoper(fd0, "-", ff0);
	/* memory free */
	xdvfree(x0);	xdvfree(x1);	xdvfree(x2);
	xdvfree(wa);	xdvfree(ff2);

	/* memory allocation */
	crf = xdvalloc(fftl);
	crf0 = xdvalloc(fftl);
	for (k = 0; k < fftl; k++) {
	    crf->data[k] = (double)k / (double)fftl * fs + (ff1->data[k] * fd->imag[k] - ff1->imag[k] * fd->data[k]) / pow(CABS(ff1->data[k], ff1->imag[k]), 2.0) * fs / PI / 2.0;
	    crf0->data[k] = (double)k / (double)fftl * fs + (ff0->data[k] * fd0->imag[k] - ff0->imag[k] * fd0->data[k]) / pow(CABS(ff0->data[k], ff0->imag[k]), 2.0) * fs / PI / 2.0;
	}

	for (k = 0; k < pif->row; k++) {
	    pif->data[k][kk] = crf->data[k] * 2.0 * PI;
	    dpif->data[k][kk] = (crf->data[k] - crf0->data[k]) * 2.0 * PI;
	    pwm->data[k][kk] = CABS(ff1->data[k], ff1->imag[k]);
	}
	
	/* memory free */
	xdvfree(ff0);	xdvfree(ff1);	xdvfree(fd);
	xdvfree(fd0);	xdvfree(crf);	xdvfree(crf0);
    }
    /* memory free */
    xdvfree(wo);    xdvfree(rr);
    xdvfree(xo);    xdvfree(lx);    

    /* memory allocation */
    slp = xdmalloc(pif->row, pif->col);
    dslp = xdmalloc(dpif->row, dpif->col);
    mmp = xdmzeros(slp->row, slp->col);
    for (k = 0; k < slp->row - 1; k++) {
	for (kk = 0; kk < slp->col; kk++) {
	    slp->data[k][kk] = (pif->data[k + 1][kk] - pif->data[k][kk]) / (fs / (double)fftl * 2.0 * PI);
	    dslp->data[k][kk] = (dpif->data[k + 1][kk] - dpif->data[k][kk]) / (2.0 * PI / (double)fftl);
	}
    }
    for (kk = 0; kk < slp->col; kk++) {
	slp->data[slp->row - 1][kk] = 0.0;
	dslp->data[slp->row - 1][kk] = 0.0;
    }

    c = znrmlcf(1);	c1 = c[0];	c2 = c[1];

    /* calculation of relative noise level */
    for (ii = 0; ii < mmp->row; ii++) {
	tmp = pow(((double)ii + 0.5) / (double)fftl * fs, 2.0);
	if (c2 < 1.7e+308 / tmp) {
	    c2 *= tmp;
	    for (k = 0; k < slp->col; k++) {
		mmp->data[ii][k] = pow(dslp->data[ii][k] / sqrt(c2), 2.0) + pow(slp->data[ii][k] / sqrt(c1), 2.0);
	    }
	} else {
	    for (k = 0; k < slp->col; k++) {
		mmp->data[ii][k] = pow(slp->data[ii][k] / sqrt(c1), 2.0);
	    }
	}
    }
    /* memory free */
    xdmfree(slp);    xdmfree(dslp);    xdmfree(dpif);

    /* Temporal smoothing */
    /* 8 ms, and odd number */
    sml = (long)round(4.0 * fs / 1000.0 / 2.0) * 2 + 1;
    /* bias due to filtering */
    smb = (sml - 1) / 2;
    /* Hanning window */
    hann = xdvalloc(sml);
    for (k = 0, sum = 0.0; k < hann->length; k++) {
	hann->data[k] = 0.5 - 0.5 * cos(2.0 * (double)(k + 1)* PI / (double)(hann->length + 1));
	sum += hann->data[k];
    }
    dvscoper(hann, "/", sum);

    /* memory allocation */
    smmp = xdmalloc(mmp->row, nfr);
    mmpvec = xdvalloc(mmp->col + sml * 2);
    /* filtering */
    for (ii = 0; ii < mmp->row; ii++) {
	for (k = 0; k < mmp->col; k++)
	    mmpvec->data[k] = mmp->data[ii][k] + 0.00001;
	for (; k < mmpvec->length; k++) mmpvec->data[k] = 0.00001;
	smmpbvec = xdvfftfiltm(hann, mmpvec, hann->length * 2);
	dvscoper(smmpbvec, "^", -1.0);
	smmpvec = xdvfftfiltm(hann, smmpbvec, hann->length * 2);
	for (k = 0; k < smmp->col; k++) {
	    smmp->data[ii][k] = 1.0 / smmpvec->data[k + sml - 2];
	}
	/* memory free */
	xdvfree(smmpbvec);	xdvfree(smmpvec);
    }
    /* memory free */
    xdvfree(mmpvec);	xdmfree(mmp);

    /* Power adaptive weighting */
    spif = xdmalloc(pif->row, nfr);
    pwm1vec = xdvalloc(pwm->col + sml * 2);
    pwm2vec = xdvalloc(pwm->col + sml * 2);
    for (ii = 0; ii < pwm->row; ii++) {
	for (k = 0; k < pwm->col; k++) {
	    pwm1vec->data[k] = pwm->data[ii][k] + 0.00001;
	    pwm2vec->data[k] = pwm->data[ii][k] * pif->data[ii][k] + 0.00001;
	}
	for (; k < pwm1vec->length; k++) {
	    pwm1vec->data[k] = 0.00001;
	    pwm2vec->data[k] = 0.00001;
	}
	spwmvec = xdvfftfiltm(hann, pwm1vec, hann->length * 2);
	spfmvec = xdvfftfiltm(hann, pwm2vec, hann->length * 2);
	for (k = 0; k < nfr; k++) {
	    spif->data[ii][k] = spfmvec->data[k + smb] / spwmvec->data[k + smb];
	}
	/* memory free */
	xdvfree(spwmvec);	xdvfree(spfmvec);
    }
    /* memory free */
    xdvfree(pwm1vec);	xdvfree(pwm2vec);
    xdvfree(hann);	xdmfree(pif);	xdmfree(pwm);
    
    /* memory allocation */
    idx = xdvalloc(f0i->length);
    if (allparam_flag != XFALSE) sum1v = xdvalloc(nfr);
    sum2v = xdvalloc(nfr);
    sum3v = xdvalloc(nfr);
    if (allparam_flag == XFALSE) {
	reff0info = xdvsalloc(1);
    } else {
	reff0info = xdvsalloc(2);
	reff0info->vector[1] = xdvalloc(nfr);
    }
    reff0info->vector[0] = xdvalloc(nfr);

    for (k = 0; k < idx->length; k++)
	idx->data[k] = MAX(0.0, f0i->data[k] / fs * (double)fftl);

    for (k = 0; k < nfr; k++) {
	if (allparam_flag != XFALSE) sum1v->data[k] = 0.0;
	sum2v->data[k] = 0.0;
	sum3v->data[k] = 0.0;
	for (ii = 0; ii < nhmx; ii++) {
	    iidx = idx->data[k] * (double)(ii + 1);
	    lidx = (long)iidx;
	    dfidx = iidx - (double)lidx;
	    vvv = (smmp->data[lidx][k] + dfidx * (smmp->data[lidx + 1][k] - smmp->data[lidx][k])) / pow((double)(ii + 1), 2.0);
	    fqv = (spif->data[lidx][k] + dfidx * (spif->data[lidx + 1][k] - spif->data[lidx][k])) / 2.0 / PI / ((double)ii + 1.0);
	    if (allparam_flag != XFALSE) sum1v->data[k] += 1.0 / vvv;
	    sum2v->data[k] += fqv / sqrt(vvv);
	    sum3v->data[k] += 1.0 / sqrt(vvv);
	}
	reff0info->vector[0]->data[k] = sum2v->data[k] / sum3v->data[k] * (f0raw->data[k] > 0.0);
	if (allparam_flag != XFALSE) reff0info->vector[1]->data[k] = sqrt(sum1v->data[k]) * (f0raw->data[k] > 0.0) + (f0raw->data[k] <= 0.0);
    }

    /* memory free */
    xdvfree(f0i);	xdvfree(idx);
    if (allparam_flag != XFALSE) xdvfree(sum1v);
    xdvfree(sum2v);	xdvfree(sum3v);
    xdmfree(smmp);	xdmfree(spif);

    return reff0info;
}

/* F0 estimation refinement */
DVECTORS refineF06(DVECTOR x,	/* input waveform */
		   double fs,	/* sampling frequency [Hz] */
		   DVECTOR f0raw,	/* F0 candidate [Hz] */
		   long fftl,	/* FFT length */
		   double eta,	/* temporal stretch factor */
		   long nhmx,	/* highest harmonic number */
		   double shiftm,	/* frame shift period [ms] */
		   long nl,	/* lower frame number */
		   long nu,	/* upper frame number */
		   XBOOL allparam_flag)	/* XFALSE*/
{
    long k, nfr, lenx, len1, len2, nlo, kk;
    long wal, bias, ii, sml, smb, lidx;
    double xlinms, shiftl, f0t, tmp, *c, c1, c2, dcl;
    double sum, dfidx, iidx, vvv, fqv;
    LVECTOR w1i = NODATA;	LVECTOR wgi = NODATA;
    DVECTOR f0i = NODATA;	DVECTOR lx = NODATA;
    DVECTOR rr = NODATA;	DVECTOR w1 = NODATA;
    DVECTOR w1b = NODATA;	DVECTOR wg = NODATA;
    DVECTOR wgg = NODATA;	DVECTOR wo = NODATA;
    DVECTOR wa = NODATA;	DVECTOR xo = NODATA;
    DVECTOR xi = NODATA;	DVECTOR x0 = NODATA;
    DVECTOR x1 = NODATA;	DVECTOR x2 = NODATA;
    DVECTOR ff0 = NODATA;	DVECTOR ff1 = NODATA;
    DVECTOR ff2 = NODATA;	DVECTOR fd = NODATA;
    DVECTOR fd0 = NODATA;	DVECTOR crf = NODATA;
    DVECTOR crf0 = NODATA;	DVECTOR hann = NODATA;
    DVECTOR mmpvec = NODATA;	DVECTOR smmpbvec = NODATA;
    DVECTOR smmpvec = NODATA;	DVECTOR pwm1vec = NODATA;
    DVECTOR pwm2vec = NODATA;	DVECTOR spwmvec = NODATA;
    DVECTOR spfmvec = NODATA;	DVECTOR idx = NODATA;
    DVECTOR sum1v = NODATA;	DVECTOR sum2v = NODATA;
    DVECTOR sum3v = NODATA;
    DVECTORS reff0info = NODATA;
    DMATRIX pif = NODATA;	DMATRIX dpif = NODATA;
    DMATRIX pwm = NODATA;	DMATRIX slp = NODATA;
    DMATRIX dslp = NODATA;	DMATRIX mmp = NODATA;
    DMATRIX smmp = NODATA;	DMATRIX spif = NODATA;

    /* memory allocation */
    f0i = xdvalloc(f0raw->length);
    for (k = 0; k < f0raw->length; k++) {
	if (f0raw->data[k] == 0.0) {
	    f0i->data[k] = 160.0;
	} else {
	    f0i->data[k] = f0raw->data[k];
	}
    }

    xlinms = (double)x->length / fs * 1000.0;
    nfr = f0i->length;
    shiftl = shiftm / 1000.0 * fs;
    lenx = x->length;
    f0t = 100.0;

    /* memory allocation */
    lx = xdvzeros(fftl * 2 + lenx);
    dvpaste(lx, x, fftl, x->length, 0);
    rr = xdvrialloc(fftl);
    for (k = 0; k < fftl; k++) {
	tmp = (double)k / (double)fftl * 2.0 * PI;
	rr->data[k] = cos(tmp);
	rr->imag[k] = sin(tmp * -1.0);
    }

    /* make window */
    w1b = xdvalloc(fftl);
    w1i = xlvalloc(fftl);
    wg = xdvalloc(fftl);
    wgi = xlvalloc(fftl);
    for (k = 0, len1 = 0, len2 = 0; k < fftl; k++) {
	tmp = ((double)k + 1.0 - (double)fftl / 2.0) / fs * f0t / eta;
	/* triangular window */
	w1b->data[k] = MAX(0.0, 1.0 - fabs(tmp));
	/* gaissian window */
	wg->data[k] = exp(-PI * pow(tmp, 2.0));
	if (w1b->data[k] > 0.0) {
	    w1i->data[len1] = k;	len1++;
	}
	if (fabs(wg->data[k]) > 0.0002) {
	    wgi->data[len2] = k;	len2++;
	}
    }
    w1 = xdvalloc(len1 + len2);
    for (k = 0; k < len1; k++) w1->data[k] = w1b->data[w1i->data[k]];
    for (; k < w1->length; k++) w1->data[k] = 0.0;
    wgg = xdvalloc(len2);
    for (k = 0; k < len2; k++) wgg->data[k] = wg->data[wgi->data[k]];
    /* convolution */
    wo = xdvfftfiltm(wgg, w1, wgg->length * 2);

    /* memory free */
    xlvfree(w1i);    xlvfree(wgi);    xdvfree(w1);
    xdvfree(w1b);    xdvfree(wg);    xdvfree(wgg);

    /* memory allocation */
    xo = xdvalloc(wo->length);
    for (k = 0; k < xo->length; k++)
	xo->data[k] = (double)k / (double)(wo->length - 1);
    nlo = wo->length - 1;

    if (nl * nu < 0) {
	nl = 1;	nu = nfr;
    }

    /* memory allocation */
    pif = xdmzeros(fftl / 2 + 1, nfr);
    dpif = xdmzeros(pif->row, nfr);
    pwm = xdmzeros(pif->row, nfr);

    for (kk = nl - 1; kk < nu; kk++) {
	if (f0i->data[kk] < 40.0) f0i->data[kk] = 40.0;
	f0t = f0i->data[kk];

	/* memoroy allocation */
	for (tmp = 0.0, len1 = 0; tmp <= 1.0;
	     tmp += f0t / (double)nlo / 100.0) len1++;
	xi = xdvalloc(len1);
	for (k = 0; k < len1; k++)
	    xi->data[k] = (double)k * f0t / (double)nlo / 100.0;
	/* interpolation of window */
	if ((wa = interp1q(xo, wo, xi)) == NODATA) {
	    printmsg(stderr, "Error: refineF02\n");
	    return NODATA;
	}
	wal = wa->length;
	/* memory free */
	xdvfree(xi);

	bias = (long)round((double)fftl - (double)wal / 2.0 + (double)kk * shiftl);
	/* truncate the sequence */
	if (wal > fftl) wal = fftl;

	/* memory allocation */
	x0 = xdvalloc(wal); x1 = xdvalloc(wal); x2 = xdvalloc(wal);
	/* multiply window */
	for (k = 0, dcl = 0.0; k < wal; k++) dcl += lx->data[k + bias];
	dcl /= (double)wal;
	for (k = 0; k < wal; k++) {
	    x0->data[k] = (lx->data[k + bias - 1] - dcl) * wa->data[k];
	    x1->data[k] = (lx->data[k + bias] - dcl) * wa->data[k];
	    x2->data[k] = (lx->data[k + bias + 1] - dcl) * wa->data[k];
	}
	/* FFT */
	ff0 = xdvfft(x0, fftl); ff1 = xdvfft(x1, fftl);	ff2 = xdvfft(x2, fftl);
	fd = xdvoper(ff2, "*", rr);	dvoper(fd, "-", ff1);
	fd0 = xdvoper(ff1, "*", rr);	dvoper(fd0, "-", ff0);
	/* memory free */
	xdvfree(x0);	xdvfree(x1);	xdvfree(x2);
	xdvfree(wa);	xdvfree(ff2);

	/* memory allocation */
	crf = xdvalloc(fftl);
	crf0 = xdvalloc(fftl);
	for (k = 0; k < fftl; k++) {
	    crf->data[k] = (double)k / (double)fftl * fs + (ff1->data[k] * fd->imag[k] - ff1->imag[k] * fd->data[k]) / pow(CABS(ff1->data[k], ff1->imag[k]), 2.0) * fs / PI / 2.0;
	    crf0->data[k] = (double)k / (double)fftl * fs + (ff0->data[k] * fd0->imag[k] - ff0->imag[k] * fd0->data[k]) / pow(CABS(ff0->data[k], ff0->imag[k]), 2.0) * fs / PI / 2.0;
	}

	for (k = 0; k < pif->row; k++) {
	    pif->data[k][kk] = crf->data[k] * 2.0 * PI;
	    dpif->data[k][kk] = (crf->data[k] - crf0->data[k]) * 2.0 * PI;
	    pwm->data[k][kk] = CABS(ff1->data[k], ff1->imag[k]);
	}
	
	/* memory free */
	xdvfree(ff0);	xdvfree(ff1);	xdvfree(fd);
	xdvfree(fd0);	xdvfree(crf);	xdvfree(crf0);
    }
    /* memory free */
    xdvfree(wo);    xdvfree(rr);
    xdvfree(xo);    xdvfree(lx);    

    /* memory allocation */
    slp = xdmalloc(pif->row, pif->col);
    dslp = xdmalloc(dpif->row, dpif->col);
    mmp = xdmzeros(slp->row, slp->col);
    for (k = 0; k < slp->row - 1; k++) {
	for (kk = 0; kk < slp->col; kk++) {
	    slp->data[k][kk] = (pif->data[k + 1][kk] - pif->data[k][kk]) / (fs / (double)fftl * 2.0 * PI);
	    dslp->data[k][kk] = (dpif->data[k + 1][kk] - dpif->data[k][kk]) / (2.0 * PI / (double)fftl);
	}
    }
    for (kk = 0; kk < slp->col; kk++) {
	slp->data[slp->row - 1][kk] = 0.0;
	dslp->data[slp->row - 1][kk] = 0.0;
    }

    c = znrmlcf(shiftm);	c1 = c[0];	c2 = c[1];

    /* calculation of relative noise level */
    for (ii = 0; ii < mmp->row; ii++) {
	tmp = pow(((double)ii + 0.5) / (double)fftl * fs, 2.0);
	if (c2 < 1.7e+308 / tmp) {
	    c2 *= tmp;
	    for (k = 0; k < slp->col; k++) {
		mmp->data[ii][k] = pow(dslp->data[ii][k] / sqrt(c2), 2.0) + pow(slp->data[ii][k] / sqrt(c1), 2.0);
	    }
	} else {
	    for (k = 0; k < slp->col; k++) {
		mmp->data[ii][k] = pow(slp->data[ii][k] / sqrt(c1), 2.0);
	    }
	}
    }
    /* memory free */
    xdmfree(slp);    xdmfree(dslp);    xdmfree(dpif);

    /* Temporal smoothing */
    /* 3 ms, and odd number */
    sml = (long)round(1.5 * fs / 1000.0 / 2.0 / shiftm) * 2 + 1;
    if (sml < 2) {
	printmsg(stderr, "Shift is too long\n");
	exit(1);
    }
    /* bias due to filtering */
    smb = (sml - 1) / 2;
    /* Hanning window */
    hann = xdvalloc(sml);
    for (k = 0, sum = 0.0; k < hann->length; k++) {
	hann->data[k] = 0.5 - 0.5 * cos(2.0 * (double)(k + 1)* PI / (double)(hann->length + 1));
	sum += SQUARE(hann->data[k]);
    }
    dvscoper(hann, "/", sum);

    /* memory allocation */
    smmp = xdmalloc(mmp->row, nfr);
    mmpvec = xdvalloc(mmp->col + sml * 2);
    /* filtering */
    for (ii = 0; ii < mmp->row; ii++) {
	for (k = 0; k < mmp->col; k++)
	    mmpvec->data[k] = mmp->data[ii][k] + 0.00001;
	for (; k < mmpvec->length; k++) mmpvec->data[k] = 0.00001;
	smmpbvec = xdvfftfiltm(hann, mmpvec, hann->length * 2);
	dvscoper(smmpbvec, "^", -1.0);
	smmpvec = xdvfftfiltm(hann, smmpbvec, hann->length * 2);
	for (k = 0; k < smmp->col; k++)
	    smmp->data[ii][k] = 1.0 / smmpvec->data[k + sml - 2];
	/* memory free */
	xdvfree(smmpbvec);	xdvfree(smmpvec);
    }
    /* memory free */
    xdvfree(mmpvec);	xdmfree(mmp);

    /* Power adaptive weighting */
    spif = xdmalloc(pif->row, nfr);
    pwm1vec = xdvalloc(pwm->col + sml * 2);
    pwm2vec = xdvalloc(pwm->col + sml * 2);
    for (ii = 0; ii < pwm->row; ii++) {
	for (k = 0; k < pwm->col; k++) {
	    pwm1vec->data[k] = pwm->data[ii][k] + 0.00001;
	    pwm2vec->data[k] = pwm->data[ii][k] * pif->data[ii][k] + 0.00001;
	}
	for (; k < pwm1vec->length; k++) {
	    pwm1vec->data[k] = 0.00001;
	    pwm2vec->data[k] = 0.00001;
	}
	spwmvec = xdvfftfiltm(hann, pwm1vec, hann->length * 2);
	spfmvec = xdvfftfiltm(hann, pwm2vec, hann->length * 2);
	for (k = 0; k < nfr; k++) {
	    spif->data[ii][k] = spfmvec->data[k + smb] / spwmvec->data[k + smb];
	}
	/* memory free */
	xdvfree(spwmvec);	xdvfree(spfmvec);
    }
    /* memory free */
    xdvfree(pwm1vec);	xdvfree(pwm2vec);
    xdvfree(hann);	xdmfree(pif);	xdmfree(pwm);
    
    /* memory allocation */
    idx = xdvalloc(f0i->length);
    if (allparam_flag != XFALSE) sum1v = xdvalloc(nfr);
    sum2v = xdvalloc(nfr);
    sum3v = xdvalloc(nfr);
    if (allparam_flag == XFALSE) {
	reff0info = xdvsalloc(1);
    } else {
	reff0info = xdvsalloc(2);
	reff0info->vector[1] = xdvalloc(nfr);
    }
    reff0info->vector[0] = xdvalloc(nfr);

    for (k = 0; k < idx->length; k++)
	idx->data[k] = MAX(0.0, f0i->data[k] / fs * (double)fftl);

    for (k = 0; k < nfr; k++) {
	if (allparam_flag != XFALSE) sum1v->data[k] = 0.0;
	sum2v->data[k] = 0.0;
	sum3v->data[k] = 0.0;
	for (ii = 0; ii < nhmx; ii++) {
	    iidx = idx->data[k] * (double)(ii + 1);
	    lidx = (long)iidx;
	    dfidx = iidx - (double)lidx;
	    vvv = (smmp->data[lidx][k] + dfidx * (smmp->data[lidx + 1][k] - smmp->data[lidx][k])) / pow((double)(ii + 1), 2.0);
	    fqv = (spif->data[lidx][k] + dfidx * (spif->data[lidx + 1][k] - spif->data[lidx][k])) / 2.0 / PI / ((double)ii + 1.0);
	    if (allparam_flag != XFALSE) sum1v->data[k] += 1.0 / vvv;
	    sum2v->data[k] += fqv / sqrt(vvv);
	    sum3v->data[k] += 1.0 / sqrt(vvv);
	}
	reff0info->vector[0]->data[k] = sum2v->data[k] / sum3v->data[k] * (f0raw->data[k] > 0.0);
	if (allparam_flag != XFALSE) reff0info->vector[1]->data[k] = sqrt(sum1v->data[k]) * (f0raw->data[k] > 0.0) + (f0raw->data[k] <= 0.0);
    }

    /* memory free */
    xdvfree(f0i);	xdvfree(idx);
    if (allparam_flag != XFALSE) xdvfree(sum1v);
    xdvfree(sum2v);	xdvfree(sum3v);
    xdmfree(smmp);	xdmfree(spif);

    return reff0info;
}

/* linear interpolation (X and XI must be monotonically increasing) */
DVECTOR interp1q(DVECTOR x, DVECTOR y, DVECTOR xi)
{
    long ki, kx, ki2;
    DVECTOR xo;

    xo = xdvalloc(xi->length);
    
    if (x->length == 1) {
	for (ki = 0; ki < xi->length; ki++) xo->data[ki] = y->data[0];
	return xo;
    }

    for (ki = 0, ki2 = 0, kx = 0; ki < xi->length; ki++) {
	for (; kx < x->length - 1; kx++) {
	    if (x->data[kx] <= xi->data[ki] &&
		xi->data[ki] <= x->data[kx + 1]) {
		/* interpolation */
		xo->data[ki] = (x->data[kx + 1] - xi->data[ki]) * y->data[kx] + (xi->data[ki] - x->data[kx]) * y->data[kx + 1];
		if (x->data[kx + 1] != x->data[kx]) {
		    xo->data[ki] /= x->data[kx + 1] - x->data[kx];
		    ki2++;
		    break;
		} else {
		    xo->data[ki] = y->data[kx];
		    ki2++;
		    break;
		}
	    }
	}
    }
    /* error check */
    if (ki2 != xo->length)
	for (ki = ki2; ki < xo->length; ki++) xo->data[ki] = xo->data[ki2 - 1];

    return xo;
}

double *znrmlcf(double f)
{
    long k;
    double n = 100.0;
    double *c;
    DVECTOR x;
    DVECTOR xx;
    DVECTOR tmp;
    DVECTOR g;
    DVECTOR dgs;

    c = xalloc(2, double);
    x = xdvalloc((long)n * 3 + 1);
    for (k = 0; k < x->length; k++)
	x->data[k] = (double)k / n;

    g = zGcBs(x, 0.0);

    dgs = xdvalloc(g->length);
    for (k = 0; k < g->length - 1; k++)
	dgs->data[k] = (g->data[k + 1] - g->data[k]) * n / 2.0 / PI / f;
    dgs->data[g->length - 1] = 0.0;

    xx = xdvscoper(x, "*", 2.0 * PI * f);
    tmp = xdvoper(xx, "*", dgs);
    dvscoper(tmp, "^", 2.0);
    c[0] = dvsum(tmp) / n;
    dvscoper(xx, "^", 2.0);
    dvoper(xx, "*", dgs);
    dvscoper(xx, "^", 2.0);
    c[1] = dvsum(xx) / n;

    /* memory free */
    xdvfree(x);
    xdvfree(xx);
    xdvfree(tmp);
    xdvfree(g);
    xdvfree(dgs);

    return c;
}

DVECTOR getf0var(DVECTOR f0raw, DVECTOR irms)
{
    long k;
    DVECTOR f0var;

    f0var = xdvalloc(irms->length);
    for (k = 0; k < irms->length; k++) {
	f0var->data[k] = pow(MAX(0.00001, irms->data[k]), 2.0);
	if (f0var->data[k] > 0.99 || f0raw->data[k] == 0.0)
	    f0var->data[k] = 100.0;
	/* 2 is a magic number. If everything is OK, it should be 1. */
    	f0var->data[k] /= 2.0;
	/* This modification is to make V/UV decision crisp */
	if (f0var->data[k] > 0.9) {
	    f0var->data[k] = 1.0;
	} else {
	    f0var->data[k] = 0.0;
	}
    }
     
    return f0var;
}

void pruningf0(DMATRIX f0v,
	       DMATRIX vrv,
	       DVECTOR f0raw,
	       double maxf0,
	       double minf0)
{
    long len;
    long k, l;
    long idx;
    long *idxv = NULL;
    double *vrvvec = NULL;

    len = MIN(f0v->col, f0raw->length);
    idxv = new long [f0v->row];
    vrvvec = new double [f0v->row];

    for (k = 0; k < len; k++) {
	if ((f0raw->data[k] > maxf0 || f0raw->data[k] < minf0)
	    && f0raw->data[k] != 0.0) {
	    for (l = 0; l < f0v->row; l++) {
		vrvvec[l] = sqrt(vrv->data[l][k]);	idxv[l] = l;
	    }
	    quicksort(vrvvec, 0, f0v->row - 1, idxv);
	    for (l = 0; l < f0v->row; l++) {
		idx = idxv[l];
		if (f0v->data[idx][k] <= maxf0 && f0v->data[idx][k] >= minf0) {
		    f0raw->data[k] = f0v->data[idx][k];
		    break;
		}
	    }
	    if (l == f0v->row) f0raw->data[k] = 0.0;
	}
    }

    delete [] idxv;
    delete [] vrvvec;

    return;
}

void plotcandf0file(DMATRIX f0v,
		    DMATRIX vrv,
		    DVECTOR f0raw,
		    char *cf0file,
		    double f0ceil,
		    double f0floor,
		    long f0shiftl)
{
    long len;
    long k, l;
    long idx;
    long *idxv = NULL;
    double *vrvvec = NULL;
    double max;
    FILE *fp;

#ifndef WIN32
    check_dir(cf0file);
#endif
    if ((fp = fopen(cf0file, "wt")) == NULL) {
	printmsg(stderr, "Can't open file: %s\n", cf0file);
	exit(1);
    }

    len = MIN(f0v->col, f0raw->length);
    idxv = new long [f0v->row];
    vrvvec = new double [f0v->row];

    fprintf(fp, "# FrameShift=%ld\n", f0shiftl);
    for (k = 0; k < len; k += f0shiftl) {
	if (f0raw->data[k] == 0.0) fprintf(fp, "0.0 100.0 ");

	for (l = 0; l < f0v->row; l++) {
	    vrvvec[l] = sqrt(vrv->data[l][k]);
	    idxv[l] = l;
	}
	quicksort(vrvvec, 0, f0v->row - 1, idxv);
	for (l = f0v->row - 1, max = 0.000001; l >= 0; l--) {
	    idx = idxv[l];
	    if (f0v->data[idx][k] <= f0ceil && f0v->data[idx][k] >= f0floor) {
		if (max < vrvvec[idx] && vrvvec[idx] != 10000.0)
		    max = vrvvec[idx];
	    }
	}
	
	for (l = 0; l < f0v->row; l++) {
	    if (f0v->data[l][k] == f0raw->data[k]) {
		fprintf(fp, "%.1f %.1f ", f0v->data[l][k], (max - vrv->data[l][k]) / max * 100.0);
		break;
	    }
	}

	for (l = 0; l < f0v->row; l++) {
	    idx = idxv[l];
	    if (f0v->data[idx][k] <= f0ceil && f0v->data[idx][k] >= f0floor) {
		if (f0v->data[idx][k] != f0raw->data[k]) {
		    if (vrvvec[l] != 10000.0) fprintf(fp, "%.1f %.1f ", f0v->data[idx][k], (max - vrvvec[l]) / max * 100.0);
		}
	    }
	}
	fprintf(fp, "\n");
    }

    delete [] idxv;
    delete [] vrvvec;

    fclose(fp);
}

void plotcandf0file_prun(DMATRIX f0v,
			 DMATRIX vrv,
			 DVECTOR f0raw,
			 char *cf0file,
			 double f0ceil,
			 double f0floor,
			 double maxf0,
			 double minf0,
			 long f0shiftl)
{
    long len;
    long k, l;
    long idx;
    long *idxv = NULL;
    double *vrvvec = NULL;
    double max;
    XBOOL write_flag = XFALSE;
    XBOOL swap_flag = XFALSE;
    FILE *fp;

#ifndef WIN32
    check_dir(cf0file);
#endif
    if ((fp = fopen(cf0file, "wt")) == NULL) {
	printmsg(stderr, "Can't open file: %s\n", cf0file);
	exit(1);
    }

    len = MIN(f0v->col, f0raw->length);
    idxv = new long [f0v->row];
    vrvvec = new double [f0v->row];

    fprintf(fp, "# FrameShift=%ld\n", f0shiftl);
    for (k = 0; k < len; k += f0shiftl) {
	write_flag = XFALSE;	swap_flag = XFALSE;
	if (f0raw->data[k] == 0.0) {
	    fprintf(fp, "0.0 100.0 ");
	    write_flag = XTRUE;
	}

	for (l = 0; l < f0v->row; l++) {
	    vrvvec[l] = sqrt(vrv->data[l][k]);
	    idxv[l] = l;
	}
	quicksort(vrvvec, 0, f0v->row - 1, idxv);
	for (l = f0v->row - 1, max = 0.000001; l >= 0; l--) {
	    idx = idxv[l];
	    if (f0v->data[idx][k] <= f0ceil && f0v->data[idx][k] >= f0floor) {
		if (max < vrvvec[idx] && vrvvec[idx] != 10000.0)
		    max = vrvvec[idx];
	    }
	}
	
	for (l = 0; l < f0v->row; l++) {
	    if (f0v->data[l][k] == f0raw->data[k]) {
		if (f0v->data[l][k] <= maxf0 && f0v->data[l][k] >= minf0) {
		    fprintf(fp, "%.1f %.1f ", f0v->data[l][k], (max - vrv->data[l][k]) / max * 100.0);
		    write_flag = XTRUE;
		} else {
		    swap_flag = XTRUE;
		}
		break;
	    }
	}

	for (l = 0; l < f0v->row; l++) {
	    idx = idxv[l];
	    if (f0v->data[idx][k] <= maxf0 && f0v->data[idx][k] >= minf0) {
		if (f0v->data[idx][k] != f0raw->data[k]) {
		    if (vrvvec[l] != 10000.0) {
			fprintf(fp, "%.1f %.1f ", f0v->data[idx][k], (max - vrvvec[l]) / max * 100.0);
			write_flag = XTRUE;
			if (swap_flag == XTRUE) {
			    f0raw->data[k] = f0v->data[idx][k];
			    swap_flag = XFALSE;
			}
		    }
		}
	    }
	}
	if (write_flag == XTRUE) {
	    fprintf(fp, "\n");
	} else {
	    fprintf(fp, "0.0 0.0\n");
	    if (swap_flag == XTRUE) f0raw->data[k] = 0.0;
	}
    }

    delete [] idxv;
    delete [] vrvvec;

    fclose(fp);
}

void plotcandf0file2(DMATRIX f0v,
		     DMATRIX vrv,
		     DVECTOR f0raw,
		     char *cf0file,
		     double f0ceil,
		     double f0floor,
		     long f0shiftl)
{
    long len;
    long k, l, m;
    long idx;
    long *idxv = NULL;
    double *vrvvec = NULL;
    double max;
    FILE *fp;

#ifndef WIN32
    check_dir(cf0file);
#endif
    if ((fp = fopen(cf0file, "wt")) == NULL) {
	printmsg(stderr, "Can't open file: %s\n", cf0file);
	exit(1);
    }

    len = MIN(f0v->col, f0raw->length);
    idxv = new long [f0v->row];
    vrvvec = new double [f0v->row];

    fprintf(fp, "# FrameShift=%ld\n", f0shiftl);
    for (k = 0, m = 0; k < len; k += f0shiftl, m++) {
	if (f0raw->data[k] == 0.0) fprintf(fp, "%ld	0.0\n", m);

	for (l = 0; l < f0v->row; l++) {
	    vrvvec[l] = sqrt(vrv->data[l][k]);
	    idxv[l] = l;
	}
	quicksort(vrvvec, 0, f0v->row - 1, idxv);
	for (l = f0v->row - 1, max = 0.000001; l >= 0; l--) {
	    idx = idxv[l];
	    if (f0v->data[idx][k] <= f0ceil && f0v->data[idx][k] >= f0floor) {
		if (max < vrvvec[idx] && vrvvec[idx] != 10000.0)
		    max = vrvvec[idx];
	    }
	}
	
	for (l = 0; l < f0v->row; l++) {
	    if (f0v->data[l][k] == f0raw->data[k]) {
		fprintf(fp, "%ld	%.1f\n", m, f0v->data[l][k]);
		break;
	    }
	}

	for (l = 0; l < f0v->row; l++) {
	    idx = idxv[l];
	    if (f0v->data[idx][k] <= f0ceil && f0v->data[idx][k] >= f0floor) {
		if (f0v->data[idx][k] != f0raw->data[k]) {
		    if (vrvvec[l] != 10000.0) fprintf(fp, "%ld	%.1f\n", m, f0v->data[idx][k]);
		}
	    }
	}
    }

    delete [] idxv;
    delete [] vrvvec;

    fclose(fp);
}

void plotcandf0file2_prun(DMATRIX f0v,
			  DMATRIX vrv,
			  DVECTOR f0raw,
			  char *cf0file,
			  double f0ceil,
			  double f0floor,
			  double maxf0,
			  double minf0,
			  long f0shiftl)
{
    long len;
    long k, l, m;
    long idx;
    long *idxv = NULL;
    double *vrvvec = NULL;
    double max;
    FILE *fp;

#ifndef WIN32
    check_dir(cf0file);
#endif
    if ((fp = fopen(cf0file, "wt")) == NULL) {
	printmsg(stderr, "Can't open file: %s\n", cf0file);
	exit(1);
    }

    len = MIN(f0v->col, f0raw->length);
    idxv = new long [f0v->row];
    vrvvec = new double [f0v->row];

    fprintf(fp, "# FrameShift=%ld\n", f0shiftl);
    for (k = 0, m = 0; k < len; k += f0shiftl, m++) {
	if (f0raw->data[k] == 0.0) fprintf(fp, "%ld	0.0\n", m);

	for (l = 0; l < f0v->row; l++) {
	    vrvvec[l] = sqrt(vrv->data[l][k]);
	    idxv[l] = l;
	}
	quicksort(vrvvec, 0, f0v->row - 1, idxv);
	for (l = f0v->row - 1, max = 0.000001; l >= 0; l--) {
	    idx = idxv[l];
	    if (f0v->data[idx][k] <= f0ceil && f0v->data[idx][k] >= f0floor) {
		if (max < vrvvec[idx] && vrvvec[idx] != 10000.0)
		    max = vrvvec[idx];
	    }
	}
	
	for (l = 0; l < f0v->row; l++) {
	    if (f0v->data[l][k] == f0raw->data[k]) {
		if (f0v->data[l][k] <= maxf0 && f0v->data[l][k] >= minf0)
		    fprintf(fp, "%ld	%.1f\n", m, f0v->data[l][k]);
		break;
	    }
	}

	for (l = 0; l < f0v->row; l++) {
	    idx = idxv[l];
	    if (f0v->data[idx][k] <= maxf0 && f0v->data[idx][k] >= minf0)
		if (f0v->data[idx][k] != f0raw->data[k])
		    if (vrvvec[l] != 10000.0) fprintf(fp, "%ld	%.1f\n", m, f0v->data[idx][k]);
	}
    }

    delete [] idxv;
    delete [] vrvvec;

    fclose(fp);
}

void writecandf0file(DVECTOR orgf0, DVECTOR extf0, char *cf0file,
		     long f0shiftl)
{
    long len;
    long k;
    FILE *fp;

#ifndef WIN32
    check_dir(cf0file);
#endif
    if ((fp = fopen(cf0file, "wt")) == NULL) {
	printmsg(stderr, "Can't open file: %s\n", cf0file);
	exit(1);
    }

    len = MIN(orgf0->length, extf0->length);

    fprintf(fp, "# FrameShift=%ld\n", f0shiftl);
    for (k = 0; k < len; k++)
	fprintf(fp, "%.1f 1 %.1f 0\n", extf0->data[k], orgf0->data[k]);
    
    fclose(fp);
}


void quicksort(double *array, long lower, long upper, long *idx)
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
	    ltmp = idx[u];
	    idx[u] = idx[l];
	    idx[l] = ltmp;
	    l++;
	    u--;
	}
    } while (l < u);
    if (lower < u) quicksort(array, lower, u, idx);
    if (l < upper) quicksort(array, l, upper, idx);
}
