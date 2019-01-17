/* straight_vconv_sub.cpp
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 *
 * Functions for Converting Features
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* functions made by Banno */
#include "defs.h"
#include "memory.h"
#include "basic.h"
#include "fileio.h"
#include "option.h"
#include "vector.h"
#include "voperate.h"
#include "matrix.h"
#include "window.h"
#include "fft.h"

#include "straight_body_sub.h"
#include "straight_vconv_sub.h"


DVECTOR xget_wave2powvec(DVECTOR xd, double fs, double frame, double shift,
			 long len, XBOOL log_flag)
{
    long framel, shiftl, framel2, pos, k, l;
    DVECTOR cx = NODATA;
    DVECTOR hwdw = NODATA;
    DVECTOR powv = NODATA;
    
    framel = (long)round(frame * fs / 1000.0);
    shiftl = (long)round(shift * fs / 1000.0);
//    printmsg(stderr, "Frame length = %f\n", (double)framel / fs * 1000.0);
//    printmsg(stderr, "shift length = %f\n", (double)shiftl / fs * 1000.0);
    // Hamming window
    hwdw = xdvalloc(framel);
    for (k = 0; k < framel; k++) hwdw->data[k] = 0.54 - 0.46 * cos(2.0 * PI * (double)k / (double)framel);
    framel2 = framel / 2;

    if (len <= 0) {
	len = 0;
	for (pos = -framel2; pos + framel2 <= xd->length; pos += shiftl) len++;
    }
    powv = xdvzeros(len);
    for (k = 0, pos = -framel2; pos + framel2 <= xd->length &&
				    k < powv->length; pos += shiftl, k++) {
	// cut signal
	cx = xdvcut(xd, pos, framel);
	// multiply hamming window
	dvoper(cx, "*", hwdw);
	// calculate power
	for (powv->data[k] = 0.0, l = 0; l < cx->length; l++)
	    powv->data[k] += cx->data[l] * cx->data[l];
	powv->data[k] /= (double)l;
	if (log_flag == XTRUE) {
	    if (powv->data[k] > 0.0) {
		powv->data[k] = log10(powv->data[k]);
	    } else {
		powv->data[k] = -5.0;
	    }
	}
	// memory free
	xdvfree(cx);
    }
    // memory free
    xdvfree(hwdw);
    
    return powv;
}


DVECTOR xget_spg2powvec(DMATRIX n2sgram, XBOOL log_flag)
{
    long k, l;
    long fftl, fftl2;
    DVECTOR powv = NODATA;

    powv = xdvalloc(n2sgram->row);
    fftl2 = n2sgram->col - 1;
    fftl = fftl2 * 2;

    for (k = 0; k < n2sgram->row; k++) {
	powv->data[k] = SQUARE(n2sgram->data[k][0]);
	for (l = 1; l < fftl2; l++)
	    powv->data[k] += 2.0 * SQUARE(n2sgram->data[k][l]);
	powv->data[k] += SQUARE(n2sgram->data[k][l]);
	powv->data[k] /= (double)fftl;
	if (log_flag == XTRUE) {
	    if (powv->data[k] != 0.0) {
		powv->data[k] = log10(powv->data[k]);
	    } else {
		powv->data[k] = -5.0;
	    }
	}
    }

    return powv;
}

double spvec2pow(DVECTOR vec, XBOOL db_flag)
{
    long k, fftl2, fftl;
    double pow;

    fftl2 = vec->length - 1;
    fftl = fftl2 * 2;
    for (k = 1, pow = SQUARE(vec->data[0]); k < fftl2; k++)
        pow += 2.0 * SQUARE(vec->data[k]);
    pow += SQUARE(vec->data[k]);
    pow /= (double)fftl;

    if (db_flag == XTRUE) pow = 10.0 * log10(pow);

    return pow;
}

DVECTOR xspg2pow_norm(DMATRIX spg)
{
    long k;
    double sum;
    DVECTOR sp = NODATA;
    DVECTOR pv = NODATA;

    /* memory allocation */
    pv = xdvalloc(spg->row);

    for (k = 0, sum = 0.0; k < spg->row; k++) {
        sp = xdmextractrow(spg, k);
        pv->data[k] = spvec2pow(sp, XFALSE);
        sum += pv->data[k];
        xdvfree(sp);
    }
    sum /= (double)k;

    for (k = 0; k < pv->length; k++)
        pv->data[k] = 10.0 * log10(pv->data[k] / sum);

    return pv;
}

/* memory alloccation (double) */
double *dalloc(int cc)
{
  double *ptr;

  if( (ptr=(double *)malloc(cc*sizeof(double))) == NULL){
    printmsg(stderr,"メモリーallocに失敗しました。\n");return(0);}
  return(ptr);
}

void fillz(double *ptr, int nitem){
    register long n;
  
    n = nitem;
    while(n--)
	*ptr++ = 0;
}

void movem(register double *a,	/* input data */
	   register double *b,	/* output data */
	   int nitem)		/* data length */
{
    register long i;

    i = nitem;
    if (a > b)
	while (i--) *b++ = *a++;
    else{
	a += i; b += i;
	while (i--) *--b = *--a;
    }
}

void freqt(double *c1,	/* minimum phase sequence (Cepstrum) */
	   long m1,	/* order of minimum phase sequence (FFT / 2) */
	   double *c2,	/* warped sequence [0]-[m2] */
	   long m2,	/* order of warped sequence */
	   double a)	/* all-pass constant */
{
    register int 	i, j;
    double		b;
    static double	*d = NULL, *g;
    static int		size;
    
    if(d == NULL){
	size = (int)m2;
	if ((d = dalloc(size+size+2)) == NULL) exit(1);
	g = d + size + 1;
    }

    if((int)m2 > size){
	free(d);
	size = (int)m2;
	if ((d = dalloc(size+size+2)) == NULL) exit(1);
	g = d + size + 1;
    }
    
    b = 1 - a*a;
    fillz(g, (int)m2+1);

    for (i=-(int)m1; i<=0; i++){
	if (0 <= (int)m2)
	    g[0] = c1[-i] + a*(d[0] = g[0]);
	if (1 <= (int)m2)
	    g[1] = b*d[0] + a*(d[1] = g[1]);
	for (j=2; j<=(int)m2; j++)
	    g[j] = d[j-1] + a*((d[j]=g[j]) - g[j-1]);
    }
    
    movem(g, c2, (int)m2+1);
}

DVECTOR xcep2mcep(DVECTOR cep,	/* Cepstrum */
		  long order,	/* Output cepstrum order */
		  long fftl,	/* FFT length */
		  XBOOL power_flag,	/* include power flag */
		  XBOOL inv_flag)	/* inverse flag, mcep->cep */
{
    //double a = 0.42;
    double a = 0.4;
    long k;
    DVECTOR cpcep = NODATA;
    DVECTOR mcep = NODATA;

    /* memory allocation */
    mcep = xdvzeros(order + 1);
    cpcep = xdvzeros(fftl / 2 + 1);
    /* minimum phase cepstrum [0]-[fftl/2] */
    if (power_flag == XTRUE) {	/* include power */
	dvcopy(cpcep, cep);
    } else {	/* don't include power */
	dvpaste(cpcep, cep, 1, cep->length, 0);
	cpcep->data[0] = 10.0;
    }
    cpcep->data[0] /= 2.0;
    cpcep->data[fftl / 2] /= 2.0;

    if (inv_flag == XTRUE) {	/* mel cep -> cep */
	freqt(cpcep->data, cpcep->length - 1, mcep->data, order, -a);
    } else {			/* cep -> mel cep */
	freqt(cpcep->data, cpcep->length - 1 , mcep->data, order, a);
    }

    xdvfree(cpcep);

    if (mcep->length > fftl / 2) {
	mcep->data[fftl / 2] *= 2.0;
    }
    if (power_flag == XTRUE) {	/* include power */
	mcep->data[0] *= 2.0;
	return mcep;
    } else {		/* don't include power */
	cpcep = xdvalloc(order);
	for (k = 0; k < order; k++) {
	    cpcep->data[k] = mcep->data[k + 1];
	}
	xdvfree(mcep);
	return cpcep;
    }
}

DVECTOR xcep2mpmcep(DVECTOR cep,	/* Cepstrum */
                    long order,	/* Output cepstrum order */
                    long fftl,	/* FFT length */
                    XBOOL power_flag,	/* include power flag */
                    XBOOL inv_flag,
                    double alpha)	/* inverse flag, mcep->cep */
{
    // double a = 0.42;
    double a = alpha;
    //   double a = 0.31;
    long k;
    DVECTOR cpcep = NODATA;
    DVECTOR mcep = NODATA;

    /* memory allocation */
    mcep = xdvzeros(order + 1);
    cpcep = xdvzeros(fftl / 2 + 1);
    /* minimum phase cepstrum [0]-[fftl/2] */
    if (power_flag == XTRUE) {	/* include power */
	dvcopy(cpcep, cep);
    } else {	/* don't include power */
	dvpaste(cpcep, cep, 1, cep->length, 0);
	cpcep->data[0] = 10.0;
    }

    if (inv_flag == XTRUE) {	/* mel cep -> cep */
	freqt(cpcep->data, cpcep->length - 1, mcep->data, order, -a);
	if (mcep->length > fftl / 2)
	    for (k = 1; k < fftl / 2; k++) mcep->data[k] /= 2.0;
	else
	    for (k = 1; k < mcep->length; k++) mcep->data[k] /= 2.0;
    } else {			/* cep -> mel cep */
	for (k = 1; k < fftl / 2; k++) cpcep->data[k] *= 2.0;
	freqt(cpcep->data, cpcep->length - 1 , mcep->data, order, a);
    }

    xdvfree(cpcep);

    if (power_flag == XTRUE) {	/* include power */
	return mcep;
    } else {		/* don't include power */
	cpcep = xdvalloc(order);
	for (k = 0; k < order; k++) cpcep->data[k] = mcep->data[k + 1];
	xdvfree(mcep);
	return cpcep;
    }
}

DVECTOR xget_spw2cep(DVECTOR spw, long order, XBOOL power_flag)
{
    long ic;
    DVECTOR spw_copy = NODATA;
    DVECTOR spw_order = NODATA;

    spw_copy = xdvclone(spw);
    
    dvscoper(spw_copy, "+", 0.000001);
    dvlog(spw_copy);
    dvifft(spw_copy);
    dvreal(spw_copy);

    /* change spw into spw_order */
    if (power_flag == XFALSE) {	/* don't include power */
	spw_order = xdvalloc(order);
	for (ic = 0; ic < order; ic++)
	    spw_order->data[ic] = spw_copy->data[ic + 1];
    } else {	/* include power */
	spw_order = xdvalloc(order + 1);
	for (ic = 0; ic < spw_order->length; ic++)
	    spw_order->data[ic] = spw_copy->data[ic];
    }

    xdvfree(spw_copy);

    return spw_order;
}

DVECTOR xget_cep2spw(DVECTOR cep, long fftl)
{
    DVECTOR spw;

    spw = xdvzeros(fftl);
    dvcopy(spw, cep);
    dvfftturn(spw);

    dvfft(spw);
    dvexp(spw);
    
    return spw;
}

DVECTOR xget_vec2cep(DVECTOR vec, long order, XBOOL power_flag)
{
    DVECTOR spw = NODATA;
    DVECTOR cep = NODATA;

    spw = xdvzeros((vec->length - 1) * 2);
    dvcopy(spw, vec);
    dvfftturn(spw);
    
    cep = xget_spw2cep(spw, order, power_flag);

    xdvfree(spw);

    return cep;
}

DVECTOR xget_cep2vec(DVECTOR cep, long fftl)
{
    DVECTOR spw = NODATA;
    DVECTOR vec = NODATA;

    spw = xget_cep2spw(cep, fftl);
    vec = xdvalloc(fftl / 2 + 1);

    dvcopy(vec, spw);

    xdvfree(spw);

    return vec;
}

DMATRIX xget_spg2cepg(DMATRIX spg, long order, XBOOL power_flag)
{
    long k;
    DVECTOR vec = NODATA;
    DVECTOR cep = NODATA;
    DMATRIX cepg = NODATA;

    /* memory allocation */
    if (power_flag == XFALSE) {
	cepg = xdmalloc(spg->row, order);
    } else {
	cepg = xdmalloc(spg->row, order + 1);
    }
    
    for (k = 0; k < spg->row; k++) {
	vec = xdmextractrow(spg, k);
	cep = xget_vec2cep(vec, order, power_flag);
	dmcopyrow(cepg, k, cep);
	xdvfree(vec);
	xdvfree(cep);
    }

    return cepg;
}

DMATRIX xget_cepg2spg(DMATRIX cepg, long fftl)
{
    long k;
    DVECTOR cep = NODATA;
    DVECTOR spw = NODATA;
    DMATRIX spg = NODATA;

    /* memory allocation */
    spg = xdmalloc(cepg->row, fftl / 2 + 1);
    
    for (k = 0; k < cepg->row; k++) {
	cep = xdmextractrow(cepg, k);
	spw = xget_cep2spw(cep, fftl);
	dmcopyrow(spg, k, spw);
	xdvfree(cep);
	xdvfree(spw);
    }

    return spg;
}

DMATRIX xcepg2mcepg(DMATRIX cepg, long order, long fftl, XBOOL power_flag,
		    XBOOL inv_flag)
{
    long k;
    DVECTOR cep = NODATA;
    DVECTOR mcep = NODATA;
    DMATRIX mcepg = NODATA;

    /* memory allocation */
    if (power_flag == XTRUE) {
	mcepg = xdmalloc(cepg->row, order + 1);
    } else {
	mcepg = xdmalloc(cepg->row, order);
    }
    
    for (k = 0; k < cepg->row; k++) {
	cep = xdmextractrow(cepg, k);
	mcep = xcep2mcep(cep, order, fftl, power_flag, inv_flag);
	dmcopyrow(mcepg, k, mcep);
	xdvfree(cep);
	xdvfree(mcep);
    }

    return mcepg;
}

DMATRIX xcepg2mpmcepg(DMATRIX cepg, long order, long fftl, XBOOL power_flag,
                      XBOOL inv_flag, double alpha)
{
    long k;
    DVECTOR cep = NODATA;
    DVECTOR mcep = NODATA;
    DMATRIX mcepg = NODATA;

    /* memory allocation */
    if (power_flag == XTRUE) {
      mcepg = xdmalloc(cepg->row, order + 1);
    } else {
      mcepg = xdmalloc(cepg->row, order);
    }
    
    for (k = 0; k < cepg->row; k++) {
      cep = xdmextractrow(cepg, k);
      mcep = xcep2mpmcep(cep, order, fftl, power_flag, inv_flag, alpha);
      dmcopyrow(mcepg, k, mcep);
      xdvfree(cep);
      xdvfree(mcep);
    }

    return mcepg;
}

DMATRIX xget_fftmat(DMATRIX mat, long fftl, long order, XBOOL inv_flag)
{
    long ri, ci;
    long hfftl;
    DVECTOR vec = NODATA;
    DMATRIX omat = NODATA;

    hfftl = fftl / 2 + 1;
    // error check
    if (mat->col > hfftl) {
	printmsg(stderr, "fft length is too short %ld\n", fftl);
	exit(1);
    }

    /* memory allocation */
    omat = xdmalloc(mat->row, order);
    vec = xdvalloc(fftl);
    
    for (ri = 0; ri < mat->row; ri++) {
	// extract vector
	for (ci = 0; ci < mat->col; ci++) vec->data[ci] = mat->data[ri][ci];
	for (; ci < hfftl; ci++) vec->data[ci] = 0.0;
	dvfftturn(vec);
	if (inv_flag == XTRUE) {	// IFFT
	    dvifft(vec);
	} else {			// FFT
	    dvfft(vec);
	}
	// copy
	dmcopyrow(omat, ri, vec);
    }
    // memory free
    xdvfree(vec);

    return omat;
}

