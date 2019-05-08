/*
 *	window.c
 *        coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "defs.h"
#include "memory.h"
#include "vector.h"
#include "window.h"

/*
 *	blackman window
 */
void blackmanf(float window[], long length)
{
    long k;
    double a, b;
    double value;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length - 1);
    b = 2.0 * a;

    for (k = 0; k < length; k++) {
	value = 0.42 - 0.5 * cos(a * (double)k) + 0.08 * cos(b * (double)k);
	window[k] = (float)value;
    }

    return;
}

void blackman(double window[], long length)
{
    long k;
    double a, b;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length - 1);
    b = 2.0 * a;

    for (k = 0; k < length; k++) {
	window[k] = 0.42 - 0.5 * cos(a * (double)k) + 0.08 * cos(b * (double)k);
    }

    return;
}

FVECTOR xfvblackman(long length)
{
    FVECTOR vec;

    vec = xfvalloc(length);
    blackmanf(vec->data, vec->length);

    return vec;
}

DVECTOR xdvblackman(long length)
{
    DVECTOR vec;

    vec = xdvalloc(length);
    blackman(vec->data, vec->length);

    return vec;
}

/*
 *	hamming window
 */
void hammingf(float window[], long length)
{
    long k;
    double a;
    double value;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length - 1);

    for (k = 0; k < length; k++) {
	value = 0.54 - 0.46 * cos(a * (double)k);
	window[k] = (float)value;
    }

    return;
}

void hamming(double window[], long length)
{
    long k;
    double a;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length - 1);

    for (k = 0; k < length; k++) {
	window[k] = 0.54 - 0.46 * cos(a * (double)k);
    }

    return;
}

FVECTOR xfvhamming(long length)
{
    FVECTOR vec;

    vec = xfvalloc(length);
    hammingf(vec->data, vec->length);

    return vec;
}

DVECTOR xdvhamming(long length)
{
    DVECTOR vec;

    vec = xdvalloc(length);
    hamming(vec->data, vec->length);

    return vec;
}

/*
 *	hanning window
 */
void hanningf(float window[], long length)
{
    long k;
    double a;
    double value;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length + 1);

    for (k = 0; k < length; k++) {
	value = 0.5 - 0.5 * cos(a * (double)(k + 1));
	window[k] = (float)value;
    }

    return;
}

void hanning(double window[], long length)
{
    long k;
    double a;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length + 1);

    for (k = 0; k < length; k++) {
	window[k] = 0.5 - 0.5 * cos(a * (double)(k + 1));
    }

    return;
}

FVECTOR xfvhanning(long length)
{
    FVECTOR vec;

    vec = xfvalloc(length);
    hanningf(vec->data, vec->length);

    return vec;
}

DVECTOR xdvhanning(long length)
{
    DVECTOR vec;

    vec = xdvalloc(length);
    hanning(vec->data, vec->length);

    return vec;
}

/*
 *	normalized blackman window
 */
void nblackmanf(float window[], long length)
{
    long k;
    double a, b;
    double value;
    double power, rms;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length - 1);
    b = 2.0 * a;

    for (k = 0, power = 0.0; k < length; k++) {
	value = 0.42 - 0.5 * cos(a * (double)k) + 0.08 * cos(b * (double)k);
	power += SQUARE(value);
	window[k] = (float)value;
    }
#if 0
    rms = sqrt(power / (double)length);
#else
    rms = sqrt(power);
#endif

    for (k = 0; k < length; k++) {
	window[k] /= (float)rms;
    }

    return;
}

void nblackman(double window[], long length)
{
    long k;
    double a, b;
    double value;
    double power, rms;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length - 1);
    b = 2.0 * a;

    for (k = 0, power = 0.0; k < length; k++) {
	value = 0.42 - 0.5 * cos(a * (double)k) + 0.08 * cos(b * (double)k);
	power += SQUARE(value);
	window[k] = value;
    }
#if 0
    rms = sqrt(power / (double)length);
#else
    rms = sqrt(power);
#endif

    for (k = 0; k < length; k++) {
	window[k] /= rms;
    }

    return;
}

FVECTOR xfvnblackman(long length)
{
    FVECTOR vec;

    vec = xfvalloc(length);
    nblackmanf(vec->data, vec->length);

    return vec;
}

DVECTOR xdvnblackman(long length)
{
    DVECTOR vec;

    vec = xdvalloc(length);
    nblackman(vec->data, vec->length);

    return vec;
}

/*
 *	hamming window
 */
void nhammingf(float window[], long length)
{
    long k;
    double a;
    double value;
    double power, rms;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length - 1);

    for (k = 0, power = 0.0; k < length; k++) {
	value = 0.54 - 0.46 * cos(a * (double)k);
	power += SQUARE(value);
	window[k] = (float)value;
    }
#if 0
    rms = sqrt(power / (double)length);
#else
    rms = sqrt(power);
#endif

    for (k = 0; k < length; k++) {
	window[k] /= (float)rms;
    }

    return;
}

void nhamming(double window[], long length)
{
    long k;
    double a;
    double value;
    double power, rms;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length - 1);

    for (k = 0, power = 0.0; k < length; k++) {
	value = 0.54 - 0.46 * cos(a * (double)k);
	power += SQUARE(value);
	window[k] = value;
    }
#if 0
    rms = sqrt(power / (double)length);
#else
    rms = sqrt(power);
#endif

    for (k = 0; k < length; k++) {
	window[k] /= rms;
    }

    return;
}

FVECTOR xfvnhamming(long length)
{
    FVECTOR vec;

    vec = xfvalloc(length);
    nhammingf(vec->data, vec->length);

    return vec;
}

DVECTOR xdvnhamming(long length)
{
    DVECTOR vec;

    vec = xdvalloc(length);
    nhamming(vec->data, vec->length);

    return vec;
}

/*
 *	hanning window
 */
void nhanningf(float window[], long length)
{
    long k;
    double a;
    double value;
    double power, rms;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length + 1);

    for (k = 0, power = 0.0; k < length; k++) {
	value = 0.5 - 0.5 * cos(a * (double)(k + 1));
	power += SQUARE(value);
	window[k] = (float)value;
    }
#if 0
    rms = sqrt(power / (double)length);
#else
    rms = sqrt(power);
#endif

    for (k = 0; k < length; k++) {
	window[k] /= (float)rms;
    }

    return;
}

void nhanning(double window[], long length)
{
    long k;
    double a;
    double value;
    double power, rms;

    if (length <= 1) {
	return;
    }
    a = 2.0 * PI / (double)(length + 1);

    for (k = 0, power = 0.0; k < length; k++) {
	value = 0.5 - 0.5 * cos(a * (double)(k + 1));
	power += SQUARE(value);
	window[k] = value;
    }
#if 0
    rms = sqrt(power / (double)length);
#else
    rms = sqrt(power);
#endif

    for (k = 0; k < length; k++) {
	window[k] /= rms;
    }

    return;
}

FVECTOR xfvnhanning(long length)
{
    FVECTOR vec;

    vec = xfvalloc(length);
    nhanningf(vec->data, vec->length);

    return vec;
}

DVECTOR xdvnhanning(long length)
{
    DVECTOR vec;

    vec = xdvalloc(length);
    nhanning(vec->data, vec->length);

    return vec;
}

/*
 *	rectangular window
 */
void nboxcarf(float window[], long length)
{
    long k;
    float value;

    if (length <= 1) {
	return;
    }

    value = (float)1.0 / (float)sqrt((double)length);
    for (k = 0; k < length; k++) {
	window[k] = value;
    }

    return;
}

void nboxcar(double window[], long length)
{
    long k;
    double value;

    if (length <= 1) {
	return;
    }

    value = 1.0 / sqrt((double)length);
    for (k = 0; k < length; k++) {
	window[k] = value;
    }

    return;
}

FVECTOR xfvnboxcar(long length)
{
    FVECTOR vec;
    
    vec = xfvalloc(length);
    nboxcarf(vec->data, vec->length);

    return vec;
}

DVECTOR xdvnboxcar(long length)
{
    DVECTOR vec;

    vec = xdvalloc(length);
    nboxcar(vec->data, vec->length);

    return vec;
}
