/*
 *	memory.c
 *        coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "defs.h"
#include "memory.h"

#define MEMORY_SERIES

char *safe_malloc(unsigned int nbytes)
{
    char *p;

    if (nbytes <= 0) {
	nbytes = 1;
    }

    p = (char *)malloc(nbytes);

    if (p == NULL) {
	printmsg(stderr, "can't malloc %d bytes\n", nbytes);
	exit(-1);
    }

    return p;
}

char *safe_realloc(char *p, unsigned int nbytes)
{
    if (nbytes <= 0) {
	nbytes = 1;
    }

    if (p == NULL) {
	return safe_malloc(nbytes);
    }

#ifdef EXIST_REALLOC_BUG
    p = (char *)realloc(p, nbytes + 1);	/* reason of +1 is realloc's bug */
#else
    p = (char *)realloc(p, nbytes);
#endif

    if (p == NULL) {
	printmsg(stderr, "can't realloc %d bytes\n",nbytes);
	exit(-1);
    }

    return p;
}

int **imatalloc(int row, int col)
{
    int i;
    int **mat;
	
    row = MAX(row, 1);
    col = MAX(col, 1);
	
    mat = xalloc(row, int *);

#ifdef MEMORY_SERIES
    *mat = xalloc(row * col, int);
    for (i = 0; i < row; i++) {
	*(mat + i) = *mat + i * col;
    }
#else 
    for (i = 0; i < row; i++) {
	*(mat + i) = xalloc(col, int);
    }
#endif

    return mat;
}

void imatfree(int **mat, int row)
{
    row = MAX(row, 1);

#ifdef MEMORY_SERIES
    xfree(*mat);
#else
    {
	int i;
	for (i = 0; i < row; i++) {
	    xfree(*(mat + i));
	}
    }
#endif

    xfree(mat);
}

short **smatalloc(int row, int col)
{
    int i;
    short **mat;

    row = MAX(row, 1);
    col = MAX(col, 1);
	
    mat = xalloc(row, short *);

#ifdef MEMORY_SERIES
    *mat = xalloc(row * col, short);
    for (i = 0; i < row; i++) {
	*(mat + i) = *mat + i * col;
    }
#else 
    for (i = 0; i < row; i++) {
	*(mat + i) = xalloc(col, short);
    }
#endif

    return mat;
}

void smatfree(short **mat, int row)
{
    row = MAX(row, 1);

#ifdef MEMORY_SERIES
    xfree(*mat);
#else
    {
	int i;
	for (i = 0; i < row; i++) {
	    xfree(*(mat + i));
	}
    }
#endif

    xfree(mat);
}

long **lmatalloc(int row, int col)
{
    int i;
    long **mat;
	
    row = MAX(row, 1);
    col = MAX(col, 1);
	
    mat = xalloc(row, long *);

#ifdef MEMORY_SERIES
    *mat = xalloc(row * col, long);
    for (i = 0; i < row; i++) {
	*(mat + i) = *mat + i * col;
    }
#else 
    for (i = 0; i < row; i++) {
	*(mat + i) = xalloc(col, long);
    }
#endif

    return mat;
}

void lmatfree(long **mat, int row)
{
    row = MAX(row, 1);

#ifdef MEMORY_SERIES
    xfree(*mat);
#else
    {
	int i;
	for (i = 0; i < row; i++) {
	    xfree(*(mat + i));
	}
    }
#endif

    xfree(mat);
}

float **fmatalloc(int row, int col)
{
    int i;
    float **mat;

    row = MAX(row, 1);
    col = MAX(col, 1);
	
    mat = xalloc(row, float *);

#ifdef MEMORY_SERIES
    *mat = xalloc(row * col, float);
    for (i = 0; i < row; i++) {
	*(mat + i) = *mat + i * col;
    }
#else 
    for (i = 0; i < row; i++) {
	*(mat + i) = xalloc(col, float);
    }
#endif

    return mat;
}

void fmatfree(float **mat, int row)
{
    row = MAX(row, 1);

#ifdef MEMORY_SERIES
    xfree(*mat);
#else
    {
	int i;
	for (i = 0; i < row; i++) {
	    xfree(*(mat + i));
	}
    }
#endif

    xfree(mat);
}

double **dmatalloc(int row, int col)
{
    int i;
    double **mat;
	
    row = MAX(row, 1);
    col = MAX(col, 1);
	
    mat = xalloc(row, double *);

#ifdef MEMORY_SERIES
    *mat = xalloc(row * col, double);
    for (i = 0; i < row; i++) {
	*(mat + i) = *mat + i * col;
    }
#else 
    for (i = 0; i < row; i++) {
	*(mat + i) = xalloc(col, double);
    }
#endif

    return mat;
}

void dmatfree(double **mat, int row)
{
    row = MAX(row, 1);

#ifdef MEMORY_SERIES
    xfree(*mat);
#else
    {
	int i;
	for (i = 0; i < row; i++) {
	    xfree(*(mat + i));
	}
    }
#endif

    xfree(mat);
}

char *strclone(char *string)
{
    char *buf;

    if (string == NULL)
	return NULL;

    buf = xalloc((strlen(string) + 1), char);
    strcpy(buf, string);

    return buf;
}
