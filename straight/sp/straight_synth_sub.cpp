/*
 *	straight_synth_sub.c : straight synthesis subroutine
 *
 *		coded by H. Banno
 *		modified by T. Toda 	2001/2/12
 *			straight_body_c03 V30k18 (matlab)
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
#include "fft.h"
#include "complex.h"
#include "filter.h"
#include "window.h"
#include "matrix.h"
#include "fileio.h"
#include "memory.h"

#include "straight_sub.h"
#include "tempo_sub.h"
#include "straight_body_sub.h"
#include "straight_synth_sub.h"
#include "straight_vconv_sub.h"


#undef POWER_NORMALIZE

extern long ss_num_frame;

/*
 *	phase table for fractional pitch
 */
DVECTOR ss_xfractpitchtbl(		/* (r): table for fractional pitch */
			  long fftl) 	/* (i): fft length */
{
    long k;
    long fftl2;
    double amp;
    double a, b, c;
    double t;
    DVECTOR phs;

    amp = 15.0;
    fftl2 = fftl / 2;

    phs = xdvalloc(fftl);
    phs->data[0] = 0.0;
    b = exp(amp);
    for (k = 1; k < fftl; k++) {
	t = (double)(k - fftl2) / (double)fftl * 2.0;
	a = exp(amp * t);
	c = t + (1.0 - a) / (1.0 + a)
	    - (1.0 + (1.0 - b) / (1.0 + b)) * t;
	phs->data[k] = c * PI;
    }

    return phs;
}

/*
 *	lowcut window
 */
DVECTOR ss_xlowcutwin(			/* (r): lowcut window */
		      double fs,	/* (i): sampling frequency */
		      double cutf,	/* (i): cutoff frequency */
		      double lowf0,	/* (i): lower f0 */
		      long fftl)	/* (i): fft length */
{
    long k;
    long hfftl;
    double value;
    DVECTOR lcw;

    hfftl = fftl / 2 + 1;

    lcw = xdvalloc(fftl);
    for (k = 0; k < hfftl; k++) {
	value = 1.0 + 
	    exp(-20.0 * ((double)k / (double)fftl * fs - lowf0) / cutf);
	lcw->data[k] = 1.0 / value;
    }
    dvfftturn(lcw);

    return lcw;
}

/*
 *	window for smoothing phase transition
 */
DVECTOR ss_xphstranswin(		/* (r): smoothing phase transition window */
			long fftl)	/* (i): fft length */
{
    long k;
    long fftl2;
    DVECTOR ptw;

    fftl2 = fftl / 2;

    ptw = xdvalloc(fftl);
    for (k = 0; k < ptw->length; k++) {
	ptw->data[k] = 1.0 / (1.0 + exp(-40.0 * (double)(k - fftl2) / (double)fftl));
    }

    return ptw;
}

/*
 *	window for smoothing group delay
 */
DVECTOR ss_xgrpdlywin(			/* (r): window for group delay */
		      double fs,	/* (i): sampling frequency */
		      double gdbw,	/* (i): group delay band width */
		      long fftl)	/* (i): fft length */
{
    long k;
    long fftl2;
    double value, sum;
    DVECTOR gdsw;	/* smoothing window for group delay */

    fftl2 = fftl / 2;

    gdsw = xdvalloc(fftl);
    /* gdbw is the equvalent rectangular band width */
    for (k = 0, sum = 0.0; k < gdsw->length; k++) {
	value = fs * (double)(k - fftl2) / (double)fftl / gdbw;
	/* slope difinition function */
	gdsw->data[k] = exp(-0.25 * PI * SQUARE(value));
	sum += gdsw->data[k];
    }
    dvscoper(gdsw, "/", sum);

    return gdsw;
}

/*
 *	window for smoothing group delay in frequency domain
 */
DVECTOR ss_xfgrpdlywin(			/* (r): gd window in frequency domain */
		       double fs,	/* (i): sampling frequency */
		       double gdbw,	/* (i): group delay band width */
		       long fftl)	/* (i): fft length */
{
    DVECTOR gdw;
    DVECTOR fgdw;

    /* gdw is the spectral smoothing window for group delay */
    gdw = ss_xgrpdlywin(fs, gdbw, fftl);

    /* fft */
    dvfftshift(gdw);
    fgdw = xdvfft(gdw, fftl);
    dvreal(fgdw);

    /* memory free */
    xdvfree(gdw);

    return fgdw;
}

/*
 *	group delay weighting function 
 */
DVECTOR ss_xgdweight(			/* (r): group delay weight */
		     double fs,		/* (i): sampling frequency */
		     double gdbw,	/* (i): group delay band width */
		     double cornf,	/* (i): lower corner frequency */
		     long fftl)		/* (i): fft length */
{
    long k;
    DVECTOR gdwt;

    gdwt = xdvalloc(fftl / 2 + 1);
    for (k = 0; k < gdwt->length; k++) {
	gdwt->data[k] = 1.0 / (1.0 + exp(-((double)(k + 1) *
					   fs / (double)fftl - cornf) / gdbw));
    }

    return gdwt;
}

/*
 *	calculate exact cepstrum from spectrum
 */
DVECTOR ss_xgetexactcep(			/* (r): exact cepstrum */
			DVECTOR pspec,		/* (i): previous spectrum */
			DVECTOR cspec,		/* (i): current spectrum */
			double fract,		/* (i): fractional part */
			long fftl)		/* (i): fft length */
{
    DVECTOR cep;
    DVECTOR tcspec;

    /* calculate exact spectrum */
    if (fract != 0.0) {
	tcspec = xdvclone(cspec);
	dvoper(tcspec, "-", pspec);
	dvscoper(tcspec, "*", fract);
	dvoper(tcspec, "+", pspec);
    } else {
	tcspec = xdvclone(pspec);
    }

    /* copy data */
    cep = xdvzeros(fftl);
    dvcopy(cep, tcspec);
    dvfftturn(cep);

    /* log of spectrum */
    dvscoper(cep, "+", 0.1);
    dvlog(cep);

    /* ifft */
    dvifft(cep);
    dvreal(cep);

    /* memory free */
    xdvfree(tcspec);

    return cep;
}

/*
 *	extract cepstrum from spectrogram
 */
DVECTOR ss_xextractcep(				/* (r): cepstrum */
		       DMATRIX n2sgram,		/* (i): smoothed spectrogram */
		       double iix,		/* (i): time index */
		       LVECTOR ftbl,		/* (i): freq stretch table */
		       long fftl)		/* (i): fft length */
{
    long ii[2];
    DVECTOR cep;
    DVECTOR spc0, spc1;
    DVECTOR sspc0, sspc1;

    ii[1] = MAX((long)ceil(iix), 0);
    ii[0] = MAX(ii[1] - 1, 0);
    ii[1] = MIN(ii[1], n2sgram->row - 1);
    ii[0] = MIN(ii[0], n2sgram->row - 1);
	
    /* calculate exact cepstrum */
    spc0 = xdmextractrow(n2sgram, ii[0]);
    spc1 = xdmextractrow(n2sgram, ii[1]);
    sspc0 = xdvremap(spc0, ftbl);
    sspc1 = xdvremap(spc1, ftbl);
    cep = ss_xgetexactcep(sspc0, sspc1, iix - (double)ii[0], fftl);

    /* memory free */
    xdvfree(spc0);
    xdvfree(spc1);
    xdvfree(sspc0);
    xdvfree(sspc1);

    return cep;
}

/*
 *	convert cepstrum to minimum phase cepstrum
 */
void ss_ceptompc(DVECTOR cep,	/* (i/o): cepstrum */
		 long fftl)	/* (i): fft length */
{
    long k;
    long fftl2;

    fftl2 = fftl / 2;
    
    for (k = 0; k < cep->length; k++) {
	if (k == 0) {
	    cep->data[k] = cep->data[k];
	} else if (k < fftl2) {
	    cep->data[k] = 2.0 * cep->data[k];
	} else {
	    cep->data[k] = 0.0;
	}
    }

    return;
}

/*
 *	calculate exact f0
 */
double ss_getexactf0(			/* (r): exact f0 */
		     double prevf0, 	/* (i): previous f0 */
		     double currf0, 	/* (i): current f0 */
		     double nextf0, 	/* (i): next f0 */
		     double fract)	/* (i): exf0 index - prevf0 index */
{
    double exf0;
    static double f0arc = 0.0;

    if (currf0 + prevf0 > 0.0) {
	/* if exist voiced frame */
	if (currf0 * prevf0 > 0.0) {
	    /* both frames are voiced */
	    exf0 = prevf0 + (currf0 - prevf0) * fract;
	    f0arc = prevf0;
	} else if (currf0 == 0.0) {
	    /*f0 = (prevf0 - f0arc) * fract + prevf0;*/
	    exf0 = prevf0;
	} else if (prevf0 == 0.0 && nextf0 > 0.0) {
	    /*f0 = (nextf0 - currf0) * (fract - 1.0) + currf0;*/
	    exf0 = currf0;
	} else {
	    exf0 = 0.0;
	}
    } else {
	exf0 = 0.0;
    }

    return exf0;
}

/*
 *	extract exact f0
 */
double ss_extractf0(			/* (r): extracted f0 */
		    DVECTOR f0l, 	/* (i): whole f0 */
		    double jjx)		/* (i): f0 index */
{
    long jj[3];
    double f0;

    jj[1] = MAX((long)ceil(jjx), 0);
    jj[0] = MAX(jj[1] - 1, 0);
    jj[2] = MAX(jj[1] + 1, 0);
    jj[0] = MIN(jj[0], f0l->length - 1);
    jj[1] = MIN(jj[1], f0l->length - 1);
    jj[2] = MIN(jj[2], f0l->length - 1);

    f0 = ss_getexactf0(f0l->data[jj[0]], f0l->data[jj[1]], f0l->data[jj[2]], 
		       jjx - (double)jj[0]);

    return f0;
}

/*
 *	get unvoiced waveform
 */
DVECTOR ss_xgetuvwave(			/* (r) : unvoiced wave */
		      DVECTOR cep,	/* (i) : cepstrum */
		      DVECTOR lcutw,	/* (i) : low cut window */
		      double period,	/* (i) : pitch period */
		      long fftl)	/* (i) : fft length */
{
    DVECTOR rs;			/* random sound source */
    DVECTOR wav;		/* waveform */
    DVECTOR uvwav;

    /* get waveform */
    wav = ss_xceptowave(cep, NODATA, lcutw, fftl);
    
    if (!SS_ALLVOICED) {
	/* get random sound source */
	rs = xdvrandn((long)period);
	dvscoper(rs, "/", sqrt(period));
	
	/* convolution random sound source */
	uvwav = xdvfftfiltm(rs, wav, fftl);

	/* memory free */
	xdvfree(rs);
    } else {
	/* copy data */
	uvwav = xdvclone(wav);
    }

    /* memory free */
    xdvfree(wav);

    return uvwav;
}

DVECTOR ss_frphstable = NODATA;		/* phase table of fractional pitch */
DVECTOR ss_phstransw = NODATA;		/* smoothing phase transition window */

/*
 *	get all pass filter for fractional pitch 
 */
DVECTOR ss_xfractpitchapf(			/* (r): apf for fractional pitch */
			  double fract,		/* (i): fractional part */
			  long fftl)		/* (i): fft length */
{
    static long last_fftl = 0;
    DVECTOR frphs;
    DVECTOR frapf;

    /* initialize */
    if (last_fftl != fftl || ss_frphstable == NODATA) {
	if (ss_frphstable != NODATA) {
	    xdvfree(ss_frphstable);
	}
	/* phs will have smooth phase function for unit delay */
	ss_frphstable = ss_xfractpitchtbl(fftl);
    }
    
    /* apf for fractional pitch */
    frphs = xdvscoper(ss_frphstable, "*", -fract);

    /* exponential of imaginary unit */
    frapf = xdvcplx(NODATA, frphs);
    dvexp(frapf);
    
    /* memory free */
    xdvfree(frphs);

    return frapf;
}

/*
 *	get noise based group delay 
 */
DVECTOR ss_xgetgrpdly(			/* (r): group delay */
		      DVECTOR fgdsw,	/* (i): gd smoothing window */
		      DVECTOR gdwt,	/* (i): group delay weight */
		      double fs,	/* (i): sampling frequency */
		      double gdbw,	/* (i): group delay band width */
		      double gdsd,	/* (i): gd standard deviation */
		      long fftl)	/* (i): fft length */
{
    long k;
    long hfftl;
    double df;
    double value;
    DVECTOR gd;			/* group delay */
    DVECTOR ngd;		/* random noise group delay */

    /* initialize */
    hfftl = fftl / 2 + 1;
    df = fs / (double)fftl * 2.0 * PI;
    
    /* noise based apf */
    ngd = xdvrandn(hfftl);

    if (gdwt != NODATA) {
	/* multiply gd weighting window */
	dvoper(ngd, "*", gdwt);
    }
    
    /* copy data */
    gd = xdvzeros(fftl);
    dvcopy(gd, ngd);

    if (fgdsw != NODATA) {
	/* turn data */
	dvfftturn(gd);

	/* convolute gd smoothing window */
	dvfft(gd);
	dvoper(gd, "*", fgdsw);
	dvifft(gd);

	/* real part */
	dvreal(gd);
    }
    
    /* adjust gd power */
    value = sqrt((double)fftl * gdbw / fs);
    value *= gdsd * df / 1000.0;
    for (k = 0; k < hfftl; k++) {
	gd->data[k] *= value;
    }
    
    /* turn data */
    dvfftturn(gd);

    /* memory free */
    xdvfree(ngd);

    return gd;
}

/*
 *	get all pass filter for random phase 
 */
DVECTOR ss_xgdtorandomapf(		/* (r): noise based apf */
			 DVECTOR gd,	/* (i): group delay */
			 long fftl)	/* (i): fft length */
{
    long k;
    double value;
    DVECTOR phs;		/* phase */
    DVECTOR apf;		/* all pass filter */

    /* initialize */
    if (ss_phstransw == NODATA) {
	ss_phstransw = ss_xphstranswin(fftl);
    }

    /* integrate group delay */
    value = gd->data[0];
    phs = xdvcumsum(gd);
    dvscoper(phs, "-", value);
    
    /* smoothing phase transition */
    value = rem(phs->data[fftl-1] + phs->data[1], 2.0 * PI) - 2.0 * PI;
    for (k = 0; k < fftl; k++) {
	phs->data[k] = -(phs->data[k] - ss_phstransw->data[k] * value);
    }
    
    /* exponential of imaginary unit */
    apf = xdvcplx(NODATA, phs);
    dvexp(apf);
    
    /* memory free */
    xdvfree(phs);

    return apf;
}

DVECTOR ss_xgetrandomapf(		/* (r): group delay */
			 DVECTOR fgdsw,	/* (i): gd smoothing window */
			 DVECTOR gdwt,	/* (i): group delay weight */
			 double fs,	/* (i): sampling frequency */
			 double gdbw,	/* (i): group delay band width */
			 double gdsd,	/* (i): gd standard deviation */
			 long fftl)	/* (i): fft length */
{
    DVECTOR gd;			/* group delay */
    DVECTOR apf;		/* all pass filter */
    
    /* get group delay */
    gd = ss_xgetgrpdly(fgdsw, gdwt, fs, gdbw, gdsd, fftl);

#if 0
    /* ROEX function (not exist in matlab source) */
    if (0) {
	long k;
	double value;

	for (k = 0; k < fftl; k++) {
	    value = FABS(gd->data[k]);
	    gd->data[k] = exp(-value) + value - 1;
	}
    }
#endif

    /* get noise based apf */
    apf = ss_xgdtorandomapf(gd, fftl);

    /* memory free */
    xdvfree(gd);

    return apf;
}

/*
 *	get one period waveform
 */
DVECTOR ss_xgetwave(			/* (r) : voiced waveform */
		    DVECTOR cep,	/* (i) : cepstrum */
		    DVECTOR lcutw,	/* (i) : low cut window */
		    DVECTOR fgdsw,	/* (i) : gd smoothing window */
		    DVECTOR gdwt,	/* (i) : group delay weight */
		    double fract,	/* (i) : fractional part */
		    double fs,	 	/* (i) : sampling frequency */
		    double gdbw,	/* (i) : group delay band width */
		    double gdsd,	/* (i) : gd standard deviation */
		    long fftl,		/* (i) : fft length */
		    int rp_flag) 	/* (i) : true if random phase */
{
    DVECTOR wav;		/* waveform */
    DVECTOR spc;		/* spectrum */

    /* calculate spectrum */
    spc = ss_xceptospec(cep, lcutw, fftl);

    /* design apf for fractional pitch */
    ss_fractpitchspec(spc, fract, fftl);

    /* design apf for random phase */
    if (rp_flag) {
	ss_randomspec(spc, fgdsw, gdwt, fs, gdbw, gdsd, fftl);
    }

    /* get waveform */
    wav = ss_xspectowave(spc, fftl);

    /* memory free */
    xdvfree(spc);

    return wav;
}

DVECTOR ss_xceptospec(			/* (r): voiced waveform */
		      DVECTOR cep,	/* (i): cepstrum */
		      DVECTOR lcutw,	/* (i): low cut window */
		      long fftl)	/* (i): fft length */
{
    DVECTOR spc;		/* spectrum */

    /* calculate spectrum */
    spc = xdvfft(cep, fftl);
    dvexp(spc);

    if (lcutw != NODATA) {
	/* multiply lowcut window */
	dvoper(spc, "*", lcutw);
    }

    return spc;
}

DVECTOR ss_xspectowave(			/* (r): voiced waveform */
		       DVECTOR spc,	/* (i): spectrum */
		       long fftl)	/* (i): fft length */
{
    DVECTOR wav;		/* waveform */

    /* get waveform */
    wav = xdvifft(spc, fftl);
    dvreal(wav);
    dvfftshift(wav);

    return wav;
}

DVECTOR ss_xceptowave(			/* (r) : voiced waveform */
		      DVECTOR cep,	/* (i) : cepstrum */
		      DVECTOR apf,	/* (i) : all pass filter */
		      DVECTOR lcutw,	/* (i) : low cut window */
		      long fftl)	/* (i) : fft length */
{
    DVECTOR wav;		/* waveform */
    DVECTOR spc;		/* spectrum */

    /* calculate spectrum */
    spc = xdvfft(cep, fftl);
    dvexp(spc);

    if (lcutw != NODATA) {
	/* multiply lowcut window */
	dvoper(spc, "*", lcutw);
    }

    if (apf != NODATA) {
	/* multiply apf */
	dvoper(spc, "*", apf);
    }

    /* get waveform */
    wav = xdvifft(spc, fftl);
    dvreal(wav);
    dvfftshift(wav);

    /* memory free */
    xdvfree(spc);

    return wav;
}

void ss_fractpitchspec(
		       DVECTOR spc,	/* (i/o): spectrum */
		       double fract,	/* (i): fractional part */
		       long fftl)	/* (i): fft length */
{
    DVECTOR frapf;		/* apf for fractional pitch */

    /* design apf for fractional pitch */
    if (fract != 0.0) {
	/* get apf for fractional pitch */
	frapf = ss_xfractpitchapf(fract, fftl);

	/* multiply apf */
	dvoper(spc, "*", frapf);

	/* memory free */
	xdvfree(frapf);
    }

    return;
}

void ss_randomspec(
		   DVECTOR spc,		/* (i/o): spectrum */
		   DVECTOR fgdsw,	/* (i): gd smoothing window */
		   DVECTOR gdwt,	/* (i): group delay weight */
		   double fs,	 	/* (i): sampling frequency */
		   double gdbw,		/* (i): group delay band width */
		   double gdsd,		/* (i): gd standard deviation */
		   long fftl)		/* (i): fft length */
{
    DVECTOR apf;		/* apf for random phase */

    /* design apf for random phase */
    apf = ss_xgetrandomapf(fgdsw, gdwt, fs, gdbw, gdsd, fftl);

    /* multiply apf */
    dvoper(spc, "*", apf);

    /* memory free */
    xdvfree(apf);

    return;
}

/*
 *	get unvoiced waveform
 */
DVECTOR ss_xspectouvwave(			/* (r): unvoiced wave */
			 DVECTOR spc,		/* (i): cepstrum */
			 double period,		/* (i): pitch period */
			 long fftl)		/* (i): fft length */
{
    DVECTOR rs;			/* random sound source */
    DVECTOR wav;		/* waveform */
    DVECTOR uvwav;

    /* get waveform */
    wav = ss_xspectowave(spc, fftl);

    if (!SS_ALLVOICED) {
	/* get random sound source */
	rs = xdvrandn((long)round(period));
	
#ifdef POWER_NORMALIZE
	if (1) {
	    long k;
	    double rspow;

	    for (k = 0, rspow = 0.0; k < rs->length; k++) {
		rspow += FABS(rs->data[k]);
	    }
	    dvscoper(rs, "/", rspow);
	}
#else
	dvscoper(rs, "/", sqrt(period));
#endif
	
	/* convolution random sound source */
	uvwav = xdvfftfiltm(rs, wav, fftl);

	/* memory free */
	xdvfree(rs);
    } else {
	/* copy data */
	uvwav = xdvclone(wav);
    }

    /* memory free */
    xdvfree(wav);

    return uvwav;
}

void ss_waveampcheck(DVECTOR wav)
{
    double value;

    value = MAX(FABS(dvmax(wav, NULL)), FABS(dvmin(wav, NULL)));
    if (value >= 32000.0) {
	printmsg(stderr, "straight synth: power is too big: %f\n", value);
	printmsg(stderr, "straight synth: execute normalization\n");
	dvscoper(wav, "*", 32000.0 / value);
    }

    return;
}


/*
 *	function for V/UV mixing version
 */
void ss_getmixrate(double f0varh, double f0varl, double *mixh, double *mixl, int u_flag)
{
    double mxh, mxl;

    if (u_flag) {
	mxh = sqrt(1.0 - 0.25 / (f0varh + 0.25));
	mxl = sqrt(1.0 - 0.25 / (f0varl + 0.25));
    } else {
	mxh = sqrt(0.25 / (f0varh + 0.25));
	mxl = sqrt(0.25 / (f0varl + 0.25));
    }

    if (mixh != NULL) {
	*mixh = mxh;
    }
    if (mixl != NULL) {
	*mixl = mxl;
    }

    return;
}

DVECTOR ss_cutwindhigh = NODATA;
DVECTOR ss_cutwindlow = NODATA;

DVECTOR ss_xgetmixweight(LVECTOR ftbl, double mixh, double mixl, 
			 double fs, long fftl)
{
    static long last_fftl = 0;
    DVECTOR chigh, clow;
    DVECTOR mixwt;

    if (last_fftl != fftl) {
	if (ss_cutwindhigh != NODATA) {
	    xdvfree(ss_cutwindhigh);
	}
	if (ss_cutwindlow != NODATA) {
	    xdvfree(ss_cutwindlow);
	}
	ss_cutwindhigh = sb_xlowcutwin(fs, 100.0, 600.0, fftl);
	ss_cutwindlow = xdvscoper(ss_cutwindhigh, "!-", 1.0);
    }

    chigh = xdvremap(ss_cutwindhigh, ftbl);
    dvscoper(chigh, "*", mixh);

    clow = xdvremap(ss_cutwindlow, ftbl);
    dvscoper(clow, "*", mixl);

    mixwt = xdvzeros(fftl);
    dvoper(mixwt, "+", chigh);
    dvoper(mixwt, "+", clow);
    dvfftturn(mixwt);
    
    /* memory free */
    xdvfree(chigh);
    xdvfree(clow);

    return mixwt;
}

DVECTOR ss_xceptospec_mix(			/* (r): voiced waveform */
			  DVECTOR cep,		/* (i): cepstrum */
			  DVECTOR lcutw,	/* (i): low cut window */
			  DVECTOR mixwt,	/* (i): mixing weight */
			  long fftl)		/* (i): fft length */
{
    DVECTOR spc;		/* spectrum */

    /* calculate spectrum */
    spc = xdvfft(cep, fftl);
    dvexp(spc);

    if (lcutw != NODATA) {
	/* multiply lowcut window */
	dvoper(spc, "*", lcutw);
    }

    if (mixwt != NODATA) {
	/* multiply mixing weight */
	dvoper(spc, "*", mixwt);
    }

    return spc;
}

DVECTOR ss_xceptowave_mix(			/* (r): voiced waveform */
			  DVECTOR cep,		/* (i): cepstrum */
			  DVECTOR apf,		/* (i): all pass filter */
			  DVECTOR lcutw,	/* (i): low cut window */
			  DVECTOR mixwt,	/* (i): mixing weight */
			  long fftl)		/* (i): fft length */
{
    DVECTOR wav;		/* waveform */
    DVECTOR spc;		/* spectrum */

    /* calculate spectrum */
    spc = ss_xceptospec_mix(cep, lcutw, mixwt, fftl);

    if (apf != NODATA) {
	/* multiply apf */
	dvoper(spc, "*", apf);
    }

    /* get waveform */
    wav = ss_xspectowave(spc, fftl);

    /* memory free */
    xdvfree(spc);

    return wav;
}

/*
 *	function for random allpass filter using f0
 */
DVECTOR ss_xgetrandomapf_f0(
			    double t0,
			    double A,
			    double B,
			    double a1,
			    double a2,
			    double p1,
			    double cornf,
			    double gdbw,
			    double fs,		/* (i): sampling frequency */
			    long fftl)		/* (i): fft length */
{
    long k;
    long hfftl;
    double ca, cb;
    DVECTOR nz;
    DVECTOR wt;
    DVECTOR lft;
    DVECTOR phs;
    DVECTOR apf;		/* all pass filter */

    /* initialize */
    hfftl = fftl / 2 + 1;
    
    /* noise based apf */
    nz = xdvrandn(hfftl);

    /* make weighting window */
    wt = xdvalloc(hfftl);
    wt->data[0] = 0.0;
    for (k = 1; k < hfftl; k++) {
	ca = ((double)k / fs) / (t0 * a1);
	ca = A * exp(-PI * SQUARE(ca));
	cb = ((double)k / fs - p1 * t0) / (t0 * a2);
	cb = B * exp(-PI * SQUARE(cb));
	wt->data[k] = ca + cb;
	/*printf("%f %f\n", ca, cb);*/
    }

#if 0
    for (k = 0; k < hfftl; k++) {
	printf("%f %f\n", wt->data[k], nz->data[k]);
    }
#endif
    
    /* multiply weighting window */
    dvoper(nz, "*", wt);

    /* copy data */
    phs =xdvzeros(fftl);
    dvcopy(phs, nz);
    
    /* FFT */
    fftturn(NULL, phs->data, fftl);
    dvfft(phs);

    /* multiply weighting function in frequency domain */
    lft = ss_xgdweight(fs, gdbw, cornf, fftl);
    dvoper(phs, "*", lft);
    dvscoper(phs, "*", -1.0);
    
    /* exponential of imaginary unit */
    apf = xdvcplx(NODATA, phs);
    dvexp(apf);
    
    /* memory free */
    xdvfree(nz);
    xdvfree(wt);
    xdvfree(lft);
    xdvfree(phs);
    
    return apf;
}

void ss_randomspec_f0(
		      DVECTOR spc,	/* (i/o): spectrum */
		      double t0,
		      double A,
		      double B,
		      double a1,
		      double a2,
		      double p1,
		      double cornf,
		      double gdbw,
		      double fs,	/* (i): sampling frequency */
		      long fftl)	/* (i): fft length */
{
    DVECTOR apf;		/* apf for random phase */

    /* design apf for random phase */
    apf = ss_xgetrandomapf_f0(t0, A, B, a1, a2, p1, cornf, gdbw, fs, fftl);
    
    /* multiply apf */
    dvoper(spc, "*", apf);

    /* memory free */
    xdvfree(apf);
    
    return;
}

/*
 *	memory free 
 */
void ss_xfree_sub(void)
{
    if (ss_frphstable != NODATA) {
	xdvfree(ss_frphstable);
	ss_frphstable = NODATA;
    }
    if (ss_phstransw != NODATA) {
	xdvfree(ss_phstransw);
	ss_phstransw = NODATA;
    }

    /* structure of V/UV mixing version */
    if (ss_cutwindhigh != NODATA) {
	xdvfree(ss_cutwindhigh);
    }
    if (ss_cutwindlow != NODATA) {
	xdvfree(ss_cutwindlow);
    }

    return;
}


/*
 *	extract cepstrum from spectrogram
 *	straight_synth_tb06	V30kr18
 */
DVECTOR ss_xextractcep_tb06(			/* (r): cepstrum */
			    DMATRIX n2sgram,	/* (i): smoothed spectrogram */
			    long iix,		/* (i): time index */
			    LVECTOR ftbl,	/* (i): freq stretch table */
			    long fftl,		/* (i): fft length */
			    DVECTOR wlcut,
			    double bias)
{
    long k, fftl2;
    DVECTOR cep;
    DVECTOR spc;
    DVECTOR sspc;

    fftl2 = fftl / 2;

    /* calculate exact cepstrum */
    spc = xdmextractrow(n2sgram, iix);
    sspc = xdvremap(spc, ftbl);

    /* log of spectrum */
    cep = xdvzeros(fftl);
    for (k = 1; k < fftl2; k++) {
	cep->data[k] = log(sspc->data[k] * wlcut->data[k] + bias);
	cep->data[fftl - k] = cep->data[k];
    }
    cep->data[0] = log(sspc->data[0] * wlcut->data[0] + bias);
    cep->data[fftl2] = log(sspc->data[fftl2] * wlcut->data[fftl2] + bias);

    /* ifft */
    dvifft(cep);
    dvreal(cep);

    /* memory free */
    xdvfree(spc);
    xdvfree(sspc);

    return cep;
}

void ss_waveampcheck_tb06(DVECTOR x,	/* signal */
			  double fs,	/* sampling frequency (Hz) */
			  double segms)       	/* segment length (ms) */
{
    long k, l, i, n, nw;
    double sum, mean, cf;
    DVECTOR x2 = NODATA;
    DVECTOR pw = NODATA;
    DVECTOR rn = NODATA;

    x2 = xdvoper(x, "*", x);
    n = (long)round(segms / 1000.0 * fs);
    nw = (long)ceil((double)x->length / (double)n);
    pw = xdvalloc(nw);
    for (k = 0, i = 0; k < nw - 1; k++) {
	for (l = 0, sum = 0.0; l < n; l++, i++) {
	    if (x2->data[i] != 0.0) {
		sum += x2->data[i];
	    } else {
		sum += 0.000001;
	    }
	}
	pw->data[k] = sum / (double)n;
    }
    for (sum = 0.0, l = 0; i < x2->length; i++, l++) {
	if (x2->data[i] != 0.0) {
	    sum += x2->data[i];
	} else {
	    sum += 0.000001;
	}
    }
    rn = xdvrandn(n - l);
    for (l = 0; l < rn->length; l++) {
	if (rn->data[l] != 0.0) {
	    sum += SQUARE(rn->data[l]) * 0.000001;
	} else {
	    sum += 0.000001;
	}
    }
    pw->data[k] = sum / (double)n;

    mean = dvmean(pw) / 30.0;
    for (k = 0, l = 0, sum = 0.0; k < pw->length; k++) {
	if (pw->data[k] > mean) {
	    sum += pw->data[k];
	    l++;
	}
    }
    cf = 20.0 * log10(32768.0) - 22.0 - 10.0 * log10(sum / (double)l);
    //cf = 20.0 * log10(25000.0) - 22.0 - 10.0 * log10(sum / (double)l);

    dvscoper(x, "*", pow(10.0, (cf / 20.0)));

    /* memory free */
    xdvfree(x2);
    xdvfree(pw);
    xdvfree(rn);
}

DMATRIX aperiodiccomp(DMATRIX dapv,
		      double ashift,
		      DVECTOR f0l,
		      double nshift)
{
    long k;
    DVECTOR x = NODATA;
    DVECTOR xi = NODATA;
    DVECTOR dapvv = NODATA;
    DVECTOR apv = NODATA;
    DMATRIX ap = NODATA;

    /* memory allocation */
    x = xdvalloc(dapv->row);
    xi = xdvalloc(f0l->length);
    ap = xdmalloc(f0l->length, dapv->col);
    for (k = 0; k < x->length; k++) x->data[k] = (double)k * ashift;
    for (k = 0; k < xi->length; k++)
	xi->data[k] = MIN(x->data[x->length - 1], (double)k * nshift);

    /* linear interpolation in time domain */
    for (k = 0; k < dapv->col; k++) {
	dapvv = xdmextractcol(dapv, k);
	apv = interp1q(x, dapvv, xi);
	dmcopycol(ap, k, apv);
	/* memory free */
	xdvfree(dapvv);
	xdvfree(apv);
    }

    /* memory free */
    xdvfree(x);
    xdvfree(xi);

    return ap;
}

DMATRIX aperiodiccomp_unit(DMATRIX dapv, double ashift, long *tinfo,
			   double nshift)
{
    long k, len, sfrm, efrm;
    DVECTOR x = NODATA;
    DVECTOR xi = NODATA;
    DVECTOR dapvv = NODATA;
    DVECTOR apv = NODATA;
    DMATRIX ap = NODATA;

    sfrm = (long)((double)tinfo[0] / nshift + 0.5);
    efrm = (long)((double)(tinfo[0] + tinfo[1]) / nshift + 0.5);
    len = efrm - sfrm;
    if (len < 1) len = 1;
    /* memory allocation */
    x = xdvalloc(dapv->row);
    xi = xdvalloc(len);
    ap = xdmalloc(len, dapv->col);
    for (k = 0; k < x->length; k++) x->data[k] = (double)k * ashift;
    for (k = 0; k < xi->length; k++)
	xi->data[k] = MIN(x->data[x->length - 1], (double)k * nshift);

    /* linear interpolation in time domain */
    for (k = 0; k < dapv->col; k++) {
	dapvv = xdmextractcol(dapv, k);
	apv = interp1q(x, dapvv, xi);
	dmcopycol(ap, k, apv);
	/* memory free */
	xdvfree(dapvv);
	xdvfree(apv);
    }

    /* memory free */
    xdvfree(x);
    xdvfree(xi);

    return ap;
}

DVECTOR makeimaptm(DMATRIX n3sgram,	/* smoothed spectrogram */
		   double f0shiftm,	/* frame shift of n3sgram (ms) */
		   double smflen,	/* frame length of spectrum movoment calculation (ms) */
		   double smfsht,	/* frame shift of spectrum movement calculation (ms) */
		   double fs,		/* sampling frequency (Hz) */
		   double tsconv)	/* time axis expansion coefficient */
{
    long k;
    double mm;
    double epsl, lalp, calp, lstp, crtp, talp, trtp;
    double smmie, smmis;
    DVECTOR sm = NODATA;
    DVECTOR smm = NODATA;
    DVECTOR smmi = NODATA;
    DVECTOR smmy = NODATA;
    DVECTOR imap = NODATA;

    mm = (double)n3sgram->row;
    
    if ((sm = spcmovedb(n3sgram, f0shiftm, smflen, smfsht)) == NODATA) {
	printmsg(stderr, "Error: makeimaptm\n");
	return NODATA;
    }

    epsl = 0.01;	lalp = 4.0;	calp = -4.0;
    for (k = 0, lstp = 0.0, crtp = 0.0; k < sm->length; k++) {
	lstp += exp(lalp / sm->data[k]);
	crtp += exp(calp / sm->data[k]);
    }

    if (tsconv * mm > lstp || tsconv * mm < crtp) return NODATA;
    
    while (fabs(crtp - tsconv * mm) / (tsconv * mm) > epsl) {
	talp = calp;
	trtp = crtp;
	calp += (lalp - calp) * (tsconv * mm - crtp) / (lstp - crtp);
	for (k = 0, crtp = 0.0; k < sm->length; k++)
	    crtp += exp(calp / sm->data[k]);
	lalp = talp;
	lstp = trtp;
    }

    /* memory allocation */
    smm = xdvalloc(sm->length);
    for (k = 0, crtp = 0.0; k < sm->length; k++) {
	crtp += exp(calp / sm->data[k]);
	smm->data[k] = crtp;
    }
    smmis = (smm->data[0] - 1.0) * f0shiftm * fs / 1000.0;
    smmie = (smm->data[smm->length - 1] - 1.0) * f0shiftm * fs / 1000.0;
    smmi = xdvalloc((long)(smmie - smmis) + 1);
    for (k = 1, smmi->data[0] = smmis; k < smmi->length; k++)
	smmi->data[k] = smmi->data[k - 1] + 1.0;
    smmy = xdvalloc(sm->length);
    for (k = 0; k < sm->length; k++) {
	smm->data[k] = (smm->data[k] - 1.0) * f0shiftm * fs / 1000.0;
	smmy->data[k] = (double)k;
    }

    /* interpola_tion */
    if ((imap = interp1lq(smm, smmy, smmi)) == NODATA) {
	printmsg(stderr, "Error: makeimaptm\n");
	exit(1);
    }

    printmsg(stderr, "         time scale function\n");
    
    /* memory free */
    xdvfree(sm);
    xdvfree(smm);
    xdvfree(smmi);
    xdvfree(smmy);

    return imap;
}

DVECTOR spcmovedb(DMATRIX n3sgram,	/* smoothed spectrogram */
		   double f0shiftm,	/* frame shift of n3sgram (ms) */
		   double smflen,	/* frame length of spectrum movoment calculation (ms) */
		   double smfsht)	/* frame shift of spectrum movement calculation (ms) */
{
    long mm, bb, ii, ib, k, l;
    double sum1, sum2;
    DVECTOR stdvec = NODATA;
    DVECTOR smo = NODATA;
    DVECTOR sm = NODATA;
    DVECTOR smx = NODATA;
    DVECTOR smxi = NODATA;
    DVECTOR smy = NODATA;
    DMATRIX dbsgram = NODATA;

    mm = n3sgram->row;
    bb = (long)round(smflen / f0shiftm);
    smo = xdvalloc((long)round((double)mm * f0shiftm / smfsht + 1.0));
    stdvec = xdvalloc(n3sgram->col);
    dbsgram = xdmalloc(n3sgram->row, n3sgram->col);
    for (k = 0; k < n3sgram->row; k++)
	for (l = 0; l < n3sgram->col; l++)
	    dbsgram->data[k][l] = 20.0 * log10(n3sgram->data[k][l]);

    ii = 0;
    while ((ib = (long)round((double)(ii + 1)* smfsht / f0shiftm)) + bb - 1
	   <= mm) {
	for (l = 0; l < n3sgram->col; l++) {
	    for (k = 0, sum1 = 0.0; k < bb; k++)
		sum1 += dbsgram->data[ib + k - 1][l];
	    sum1 /= (double)bb;
	    for (k = 0, sum2 = 0.0; k < bb; k++)
		sum2 += SQUARE(dbsgram->data[ib + k - 1][l] - sum1);
	    stdvec->data[l] = sqrt(sum2 / (double)(bb - 1));
	}
	smo->data[ii] = dvmean(stdvec);
	ii++;
    }

    if (ii == 0) {	/* matlab error */
	printmsg(stderr, "Error: spcmovedb\n");
	return NODATA;
    }

    /* memory free */
    xdvfree(stdvec);
    xdmfree(dbsgram);

    smx = xdvalloc(ii + 2);
    smy = xdvalloc(smx->length);
    for (k = 1; k < smx->length - 1; k++)
	smx->data[k] = ((double)k - 0.5) * smfsht;
    smx->data[0] = 1.0;			smx->data[k] = mm * f0shiftm;
    dvpaste(smy, smo, 1, ii, 0);
    smy->data[0] = smo->data[0];	smy->data[k] = smo->data[ii - 1];
    
    smxi = xdvalloc(mm);
    for (k = 0; k < mm; k++) smxi->data[k] = (double)(k + 1) * f0shiftm;

    if ((sm = interp1q(smx, smy, smxi)) == NODATA) {
	printmsg(stderr, "Error: spcmovedb\n");
	return NODATA;
    }

    /* memory free */
    xdvfree(smo);
    xdvfree(smx);
    xdvfree(smy);
    xdvfree(smxi);

    return sm;
}
