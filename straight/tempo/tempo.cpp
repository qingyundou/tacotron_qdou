/*
 *      tempo.c : F0 Extraction
 *	V30k18 (matlab)
 *
 *	      coded by T. Toda                2001/2/6
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

#include "straight_sub.h"
#include "tempo_sub.h"

#include "tempo.h"

DVECTORS tempo(DVECTOR x,
	       double maxf0,		/* 800.0 */
	       double minf0,		/* 40.0 */
	       double f0ceil,		/* 800.0 */
	       double f0floor,		/* 40.0 */
	       double fs,		/* 16000.0 */
	       double shiftm,		/* 1.0 */
	       double f0shiftm,      	/* 1.0 */
	       char *cf0file,
	       XBOOL f0var_flag,
	       XBOOL allparam_flag,	/* XFALSE */
	       XBOOL msg_flag,
	       XBOOL onlycf0_flag)
{
    long nvo, nvc, k, len, f0jmp = 1;
    double dn, cpf0shiftm;
    DVECTOR dsx = NODATA;
    DVECTOR f0var = NODATA;
    DVECTORS pwinfo = NODATA;
    DVECTORS f0trinfo = NODATA;
    DVECTORS reff0info = NODATA;
    DVECTORS reff0info2 = NODATA;
    DMATRICES f0infomat = NODATA;

    nvo = 24;
    nvc = (long)ceil(log(f0ceil/f0floor) / log(2.0) * (double)nvo);

    // f0shiftm -> shiftm
    cpf0shiftm = f0shiftm;
    if (f0shiftm < shiftm) {
	f0shiftm = shiftm;	f0jmp = 1;
	if (cpf0shiftm != f0shiftm)
	    fprintf(stderr, "f0shift %4.1f[ms]\n", f0shiftm);
    } else {
	f0jmp = (long)(f0shiftm / shiftm);
	if ((double)f0jmp != f0shiftm / shiftm)
	    f0shiftm = shiftm * (double)f0jmp;
	if (cpf0shiftm != f0shiftm)
	    fprintf(stderr, "f0shift %4.1f[ms]\n", f0shiftm);
    }

    /* f0infomat->matrix[0]: f0v, [1]: vrv, ([2]: dfv, [3]: aav, [4]nf) */
    f0infomat = fixpF0VexMltpBG4(x, fs, f0floor, nvc, nvo, 1.2, shiftm,
				 1.0, 5, 0.5, 1, allparam_flag);

    /* pwinfo->vector[0]: pwt, [1]: pwh */
    pwinfo = plotcpower(x, fs, shiftm);
    
    /* f0trinfo->vector[0]: f0raw, [1]: irms, ([2]: df, [3]: amp) */
    f0trinfo = f0track5(f0infomat, pwinfo, shiftm, allparam_flag);

    if (msg_flag == XTRUE)
	fprintf(stderr, "Max F0 = %f, Min F0 = %f\n", maxf0, minf0);
    if(!strnone(cf0file)) {
	if (msg_flag == XTRUE)
	    fprintf(stderr, "candidate F0 file: %s\n", cf0file);
//	plotcandf0file(f0infomat->matrix[0], f0infomat->matrix[1],
//		       f0trinfo->vector[0], cf0file, f0ceil, f0floor, f0jmp);
	plotcandf0file_prun(f0infomat->matrix[0], f0infomat->matrix[1],
			    f0trinfo->vector[0], cf0file, f0ceil, f0floor,
			    maxf0, minf0, f0jmp);
	if (onlycf0_flag == XTRUE) {
	    /* memory free */
	    xdvsfree(pwinfo);
	    xdmsfree(f0infomat);
	    xdvsfree(f0trinfo);
	    
	    if (msg_flag == XTRUE)
		fprintf(stderr, "done\n");
	    exit(0);
	}
    } else {
	pruningf0(f0infomat->matrix[0], f0infomat->matrix[1],
		  f0trinfo->vector[0], maxf0, minf0);
    }
    /* memory free */
    xdvsfree(pwinfo);
    xdmsfree(f0infomat);

    /* down sampling */
    dn = floor(fs / (800.0 * 3.0 * 2.0));
    dsx = decimate(x, (long)dn);

    /* reff0info->vector[0]: f0raw, ([1]: ecr) */
    reff0info = refineF06(dsx, fs / dn, f0trinfo->vector[0], 512, 1.1, 3,
			  shiftm, 1, f0trinfo->vector[0]->length,
			  allparam_flag);

    reff0info2 = xdvsalloc(2);
//    reff0info2->vector[0] = xdvclone(reff0info->vector[0]);
    len = (long)ceil((double)reff0info->vector[0]->length / (double)f0jmp);
    reff0info2->vector[0] = xdvalloc(len);
    for (k = 0; k < len; k++)
	reff0info2->vector[0]->data[k] = reff0info->vector[0]->data[k * f0jmp];

    if (f0var_flag != XFALSE) {
	f0var = getf0var(reff0info->vector[0], f0trinfo->vector[1]);
	reff0info2->vector[1] = xdvclone(f0var);

	/* memory free */
	xdvfree(f0var);
    }

    /* memory free */
    xdvfree(dsx);
    xdvsfree(f0trinfo);
    xdvsfree(reff0info);
    
    return reff0info2;
}


/* Fixed point analysis to extract F0*/
DMATRICES fixpF0VexMltpBG4(DVECTOR x,	/* input signal */
			   double fs,	/* sampling frequency [Hz] */
			   double f0floor,/* lowest frequency for F0 search */
			   long nvc,	/* total number of filter channels */
			   long nvo,	/* number of channels per octave */
			   double mu,	/* temporal stretching factor */
			   double shiftm,/* frame shift in ms */
			   double smp,	/* smoothing length relative to fc (ratio) */
			   double minm,	/* minimum smoothing length [ms] */
			   double pc,	/* exponent to represent nonlinear summation */
			   int nc,	/* number of harmonic component to use (1, 2, 3) */
			   XBOOL allparam_flag)	/* XFALSE */
{
    long k, rn, cn, mm, ii, idx, np;
    double fxh, dn, pm1pc, pm2pc, pm3pc, step;
/*    double pm1amp, pm1namp;*/
    double *c, c1, c2b, c2, kd;
    LVECTOR ki = NODATA;
    LVECTOR nf = NODATA;
    DVECTOR fxx = NODATA;
    DVECTOR dsx = NODATA;
    DVECTOR pif2v = NODATA;
    DVECTOR smapv = NODATA;
    DVECTOR dpifv = NODATA;
    DVECTOR pm1v = NODATA;
    DVECTORS f0info = NODATA;
    DMATRIX pm1 = NODATA;
    DMATRIX pm2 = NODATA;
    DMATRIX pm3 = NODATA;
    DMATRIX pmb2 = NODATA;
    DMATRIX pmb3 = NODATA;
    DMATRIX pif1 = NODATA;
    DMATRIX pif2 = NODATA;
    DMATRIX pif3 = NODATA;
    DMATRIX pifb2 = NODATA;
    DMATRIX pifb3 = NODATA;
    DMATRIX dpif = NODATA;
/*    DMATRIX damp = NODATA;*/
    DMATRIX mmp = NODATA;
    DMATRIX smap = NODATA;
    DMATRIX fixpp = NODATA;
    DMATRIX fixvv = NODATA;
    DMATRIX fixdf = NODATA;
    DMATRIX fixav = NODATA;
    DMATRICES slppbl = NODATA;
    DMATRICES dslpdpbl = NODATA;
    DMATRICES f0infomat = NODATA;

    /* cut low noise (f < f0floor) */
    cleaninglownoise(x, fs, f0floor);

    /* calculate the frequency of the (nvc)-th filter */
    fxx = xdvalloc(nvc);
    for (k = 0, fxh = 0.0; k < fxx->length; k++) {
	fxx->data[k] = f0floor * pow(2.0, (double)k / (double)nvo);
	if (fxh < fxx->data[k]) fxh = fxx->data[k];
    }

    /* number of decimation */
    dn = MAX(1.0, floor(fs / (fxh * 6.3)));

    if (nc > 1) {
	/* decimation */
	dsx = decimate(x, (long)dn);

	if (nc > 2) {
	    pmb3 = multanalytFineCSPB(dsx, fs / dn, f0floor, nvc, nvo, mu, 3);
	    pifb3 = zwvlt2ifq(pmb3, fs / dn);
	    pm3 = xdmrialloc(pmb3->row,  (long)ceil((double)pmb3->col / 3.0));
	    pif3 = xdmalloc(pm3->row,  pm3->col);
	    for (cn = 0; cn < pm3->col; cn++) {
		for (rn = 0; rn < pm3->row; rn++) {
		    pif3->data[rn][cn] = pifb3->data[rn][cn * 3];
		    pm3->data[rn][cn] = pmb3->data[rn][cn * 3];
		    pm3->imag[rn][cn] = pmb3->imag[rn][cn * 3];
		}
	    }
	    /* memory free */
	    xdmfree(pmb3);
	    xdmfree(pifb3);
	}
	if (nc > 1) {
	    pmb2 = multanalytFineCSPB(dsx, fs / dn, f0floor, nvc, nvo, mu, 2);
	    pifb2 = zwvlt2ifq(pmb2, fs / dn);
	    pm2 = xdmrialloc(pmb2->row,  (long)ceil((double)pmb2->col / 3.0));
	    pif2 = xdmalloc(pm2->row,  pm2->col);
	    for (cn = 0; cn < pm2->col; cn++) {
		for (rn = 0; rn < pm2->row; rn++) {
		    pif2->data[rn][cn] = pifb2->data[rn][cn * 3];
		    pm2->data[rn][cn] = pmb2->data[rn][cn * 3];
		    pm2->imag[rn][cn] = pmb2->imag[rn][cn * 3];
		}
	    }
	    /* memory free */
	    xdmfree(pmb2);
	    xdmfree(pifb2);
	}
	/* memory free */
	xdvfree(dsx);
    }
    /* decimation */
    dsx = decimate(x, (long)dn * 3);

    pm1 = multanalytFineCSPB(dsx, fs / dn / 3.0, f0floor, nvc, nvo, mu, 1);
    pif1 = zwvlt2ifq(pm1, fs / dn / 3.0);
    /* memory free */
    xdvfree(dsx);

    mm = pif1->col;
    if (nc > 1) mm = MIN(mm, pif2->col);
    if (nc > 2) mm = MIN(mm, pif3->col);

    if (nc == 2) {
	for (ii = 0; ii < mm; ii++) {
	    for (k = 0; k < pif2->row; k++) {
		pm1pc = pow(CABS(pm1->data[k][ii], pm1->imag[k][ii]), pc);
		pm2pc = pow(CABS(pm2->data[k][ii], pm2->imag[k][ii]), pc);

		pif2->data[k][ii] = (pif1->data[k][ii] * pm1pc + pif2->data[k][ii] / 2.0 * pm2pc) / (pm1pc + pm2pc);
	    }
	}
	/* memory free */
	xdmfree(pm2);
    }
    if (nc == 3) {
	for (ii = 0; ii < mm; ii++) {
	    for(k = 0; k < pif2->row; k++) {
		pm1pc = pow(CABS(pm1->data[k][ii], pm1->imag[k][ii]), pc);
		pm2pc = pow(CABS(pm2->data[k][ii], pm2->imag[k][ii]), pc);
		pm3pc = pow(CABS(pm3->data[k][ii], pm3->imag[k][ii]), pc);

		pif2->data[k][ii] = (pif1->data[k][ii] * pm1pc + pif2->data[k][ii] / 2.0 * pm2pc + pif3->data[k][ii] / 3.0 * pm3pc) / (pm1pc + pm2pc + pm3pc);
	    }
	}
	/* memory free */
	xdmfree(pm2);
	xdmfree(pm3);
	xdmfree(pif3);
    }
    if (nc == 1) pif2 = ss_xdmclone(pif1);
    /* memory free */
    xdmfree(pif1);

    for (rn = 0; rn < pif2->row; rn++) {
	for (cn = 0; cn < pif2->col; cn++) {
	    pif2->data[rn][cn] *= 2.0 * PI;
	}
    }
    dn *= 3.0;

    slppbl = zifq2gpm2(pif2, f0floor, nvo);

    dpif = xdmalloc(pif2->row, pif2->col);
    for (cn = 0; cn < dpif->col - 1; cn++) {
	for (rn = 0; rn < dpif->row; rn++) {
	    dpif->data[rn][cn]
		= (pif2->data[rn][cn + 1] - pif2->data[rn][cn]) * fs / dn;
	}
    }
    for (rn = 0; rn < dpif->row; rn++)
	dpif->data[rn][dpif->col - 1] = dpif->data[rn][dpif->col - 2];
    dslpdpbl = zifq2gpm2(dpif, f0floor, nvo);

/*    damp = xdmalloc(pm1->row, pm1->col);
    for (cn = 0; cn < damp->col - 1; cn++) {
	for (rn = 0; rn < damp->row; rn++) {
	    pm1namp = CABS(pm1->data[rn][cn + 1], pm1->imag[rn][cn + 1]);
	    pm1amp = CABS(pm1->data[rn][cn], pm1->imag[rn][cn]);
	    damp->data[rn][cn] = (pm1namp - pm1amp) * fs / dn;
	    damp->data[rn][cn] /= pm1amp;
	}
    }
    for (rn = 0; rn < damp->row; rn++) {
	pm1namp = CABS(pm1->data[rn][damp->col - 1], pm1->imag[rn][damp->col - 1]);
	pm1amp = CABS(pm1->data[rn][damp->col - 2], pm1->imag[rn][damp->col - 2]);
	damp->data[rn][damp->col - 1] = (pm1namp - pm1amp) * fs / dn;
	damp->data[rn][cn] /= pm1namp;
    }*/

    for (k = 0; k < fxx->length; k++) fxx->data[k] *= 2.0 * PI;

    mmp = xdmzeros(dslpdpbl->matrix[0]->row, dslpdpbl->matrix[0]->col);

    c = znrmlcf2(1);	c1 = c[0];	c2b = c[1];
    free((char *)(c));	c = NULL;

    for (ii = 0, c1 = sqrt(c1); ii < pif2->row; ii++) {
	c2 = c2b * pow(fxx->data[ii] / 2.0 / PI, 2.0);
	c2 = sqrt(c2);
	for (cn = 0; cn < mmp->col; cn++)
	    mmp->data[ii][cn] = pow(dslpdpbl->matrix[0]->data[ii][cn] / c2, 2.0) + pow(slppbl->matrix[0]->data[ii][cn] / c1, 2.0);
    }

    if (smp != 0.0) {
	smap = zsmoothmapB(mmp, fs / dn, f0floor, nvo, smp, minm, 0.4);
    } else {
	smap = ss_xdmclone(mmp);
    }
    /* memory free */
    xdmfree(mmp);
    xdmsfree(slppbl);
    xdmsfree(dslpdpbl);

    /* memory allocation */
    fixpp = xdmalloc((long)round((double)pif2->row / 3.0), pif2->col);
    fixvv = xdmalloc(fixpp->row, fixpp->col);
    for (rn = 0; rn < fixpp->row; rn++) {
	for (cn = 0; cn < fixpp->col; cn++) {
	    fixpp->data[rn][cn] = 1000000.0;
	    fixvv->data[rn][cn] = 100000000.0;
	}
    }
    if (allparam_flag != XFALSE) {
	fixdf = xdmalloc(fixpp->row, fixpp->col);
	fixav = xdmalloc(fixpp->row, fixpp->col);
	for (rn = 0; rn < fixpp->row; rn++) {
	    for (cn = 0; cn < fixpp->col; cn++) {
		fixdf->data[rn][cn] = 100000000.0;
		fixav->data[rn][cn] = 1000000000.0;
	    }
	}
    }
    nf = xlvzeros(fixpp->col);

    for (ii = 0; ii < fixpp->col; ii++) {
	pif2v = xdmextractcol(pif2, ii);
	smapv = xdmextractcol(smap, ii);
	if (allparam_flag != XFALSE) {
	    dpifv = xdmextractcol(dpif, ii);
	    pm1v = xdmextractcol(pm1, ii);
	    dvscoper(dpifv, "/", 2.0 * PI);
	}
	
	f0info = zfixpfreq3(fxx, pif2v, smapv, dpifv, pm1v, allparam_flag);

	for (k = 0; k < f0info->vector[0]->length; k++) {
	    /* fixed point frequency vector */
	    fixpp->data[k][ii] = f0info->vector[0]->data[k];
	    /* relative interfering energy vector */
	    fixvv->data[k][ii] = f0info->vector[1]->data[k];
	    if (allparam_flag != XFALSE) {
		/* fixed point slope vector */
		fixdf->data[k][ii] = f0info->vector[2]->data[k];
		/* amplitude list for fixed points */
		fixav->data[k][ii] = f0info->vector[3]->data[k];
	    }
	    nf->data[ii] = k;
	    if (fixpp->data[k][ii] == 0.0) fixpp->data[k][ii] = 1000000.0;
	}
	/* memory free */
	xdvfree(pif2v);
	xdvfree(smapv);
	xdvsfree(f0info);
	if (allparam_flag != XFALSE) {
	    xdvfree(dpifv);
	    xdvfree(pm1v);
	}
    }
    /* memory free */
    xdvfree(fxx);
    xdmfree(pif2);
    xdmfree(dpif);
    xdmfree(pm1);

    np = lvmax(nf, NULL);	np++;

    /* frame update index */
//    ki = xlvalloc(fixpp->col);
    step = shiftm / dn * fs / 1000.0;
    ki = xlvalloc((long)((double)fixpp->col / step) + 1);
    for (kd = 0.0, ki->data[0] = 0, mm = 0;
	 kd <= (double)(fixpp->col - 1); kd += step) {
	k = (long)round(kd);
//	if (ki->data[mm - 1] < k) {
	ki->data[mm] = k;
	mm++;
//	}
    }

    /* memory allocation */
    if (allparam_flag == XFALSE) {
	f0infomat = xdmsalloc(2);
    } else {
	f0infomat = xdmsalloc(5);
	f0infomat->matrix[2] = xdmalloc(np, mm);
	f0infomat->matrix[3] = xdmalloc(np, mm);
	f0infomat->matrix[4] = xdmalloc(1, mm);
    }
    f0infomat->matrix[0] = xdmalloc(np, mm);
    f0infomat->matrix[1] = xdmalloc(np, mm);

    for (k = 0; k < mm; k++) {
	idx = ki->data[k];
	for (rn = 0; rn < np; rn++) {
	    /* fixed point frequency vector */
	    f0infomat->matrix[0]->data[rn][k]
		= fixpp->data[rn][idx] / 2.0 / PI;
	    /* relative interfering energy vector */
	    f0infomat->matrix[1]->data[rn][k] = fixvv->data[rn][idx];
	    if (allparam_flag != XFALSE) {
		/* fixed point slope vector */
		f0infomat->matrix[2]->data[rn][k] = fixdf->data[rn][idx];
		/* amplitude list for fixed points */
		f0infomat->matrix[3]->data[rn][k] = fixav->data[rn][idx];
	    }
	}
	if (allparam_flag != XFALSE)
	    f0infomat->matrix[4]->data[0][k] = (double)nf->data[idx];
    }
    
    /* memory free */
    xlvfree(ki);
    xlvfree(nf);
    xdmfree(fixpp);
    xdmfree(fixvv);
    if (allparam_flag != XFALSE) {
	xdmfree(fixdf);
	xdmfree(fixav);
    }
    xdmfree(smap);
    
    return f0infomat;
}
