/*
 *	kaiser.c
 *        coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "defs.h"
#include "kaiser.h"

#define MIN_TRANS 0.0001

void getkaiserparam(double sidelobe, double trans, double *beta, long *length)
{
    double value;

    if (sidelobe < 21) {
	*beta = 0;
    } else if (sidelobe > 50) {
	*beta = 0.1102 * ((double)sidelobe - 8.7);
    } else {
	value = (double)sidelobe - 21.0;
	*beta = 0.5842 * pow(value, 0.4) + 0.07886 * value;
    }

    if (trans == 0.0) trans = MIN_TRANS;
    *length = (long)(((double)sidelobe - 8.0) / (2.285 * PI * trans));
    
    return;
}

int kaiser_org(double w[], long n, double beta)
{
    double an1, t, rms;
    long i;

    if (n <= 1) 
	return FAILURE;
    rms = 0.0;
    an1 = 1.0 / (double)(n - 1);
    for(i = 0; i < n; ++i) {
	t = ((double)( i + i - n + 1 )) * an1;
	w[i] = ai0((double)(beta * sqrt(1.0 - t * t)));
	rms += w[i] * w[i];
    }

    /* Normalize w[i] to have a unity power gain. */
    rms = sqrt((double)n / rms);
    while(n-- > 0) *w++ *= rms;

    return SUCCESS;
}

/* This function is buggy. */
double ai0_org(double x)
{
    int i;
    double y, e, de, sde, t;

    y = x / 2.0;
    t = 1.0e-12;
    e = 1.0;
    de = 1.0;
    for (i = 1; i <= 100; i++) {
        de *= y / (double)i;
        sde = de * de;
        e += sde;
        if (sde < e * t)
            break;
    }

    return e;
}

int kaiser(double w[], long n, double beta)
{
    double an1, t;
    long i;

    if (n <= 1) 
	return FAILURE;

    an1 = 1.0 / (double)(n - 1);
    for(i = 0; i < n; i++) {
	t = ((double)(2 * i - n + 1)) * an1;
	w[i] = ai0(beta * sqrt(1.0 - t * t));
	w[i] /= ai0(beta);
    }

    return SUCCESS;
}

double ai0(double x)
{
    double d, ds, s;

    ds = 1;
    d = 2;
    s = ds;
    ds = (x * x) / 4.0;
    while (ds >= 0.2e-8*s) {
	d += 2;
	s += ds;
	ds *= (x * x) / (d * d);
    }
    
    return s;
}
