/*
 *	straight_sub.c : straight subroutine
 *
 *	2001/2/12	coded by T. Toda
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NR_END 1
#define FREE_ARG char*

/* functions made by Banno */
#include "defs.h"
#include "fileio.h"
#include "voperate.h"
#include "memory.h"
#include "option.h"

#include "straight_sub.h"

void ss_nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	printmsg(stderr,"Numerical Recipes run-time error...\n");
	printmsg(stderr,"%s\n",error_text);
	printmsg(stderr,"...now exiting to system...\n");
	exit(1);
}

double *ss_dvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) ss_nrerror("allocation failure in dvector()");
	return v-nl+NR_END;
}

void ss_free_dvector(double *v, long nl)
/* free a double vector allocated with dvector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void ss_svdcmp(double **a, long m, long n, double w[], double **v)
{
    static double dmaxarg1,dmaxarg2;
    static int iminarg1,iminarg2;
  double ss_pythag(double a, double b);
  int flag,i,its,j,jj,k;
    int l = 0, nm = 0;
  double anorm,c,f,g,h,s,scale,x,y,z,*rv1;
   
  rv1=ss_dvector(1,n);
  g=scale=anorm=0.0;
  for (i=1;i<=n;i++) {
    l=i+1;
    rv1[i]=scale*g;
    g=s=scale=0.0;
    if (i <= m) {
      for (k=i;k<=m;k++) scale += fabs(a[k][i]);
      if (scale) {
	for (k=i;k<=m;k++) {
	  a[k][i] /= scale;
	  s += a[k][i]*a[k][i];
	}
	f=a[i][i];
	g = -ss_SIGN(sqrt(s),f);
	h=f*g-s;
	a[i][i]=f-g;
	for (j=l;j<=n;j++) {
	  for (s=0.0,k=i;k<=m;k++) s += a[k][i]*a[k][j];
	  
	  f=s/h;
	  for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
	}
	for (k=i;k<=m;k++) a[k][i] *= scale;
      }
    }
    w[i]=scale *g;
    g=s=scale=0.0;
    if (i <= m && i != n) {
      for (k=l;k<=n;k++) scale += fabs(a[i][k]);
      if (scale) {
	for (k=l;k<=n;k++) {
	  a[i][k] /= scale;
	  s += a[i][k]*a[i][k];
	}
	f=a[i][l];
	g = -ss_SIGN(sqrt(s),f);
	h=f*g-s;
	a[i][l]=f-g;
	for (k=l;k<=n;k++) rv1[k]=a[i][k]/h;
	for (j=l;j<=m;j++) {
	  for (s=0.0,k=l;k<=n;k++) s += a[j][k]*a[i][k];
	  for (k=l;k<=n;k++) a[j][k] += s*rv1[k];
	  
	}
	for (k=l;k<=n;k++) a[i][k] *= scale;
      }
    }
    anorm=ss_DMAX(anorm,(fabs(w[i])+fabs(rv1[i])));
  }
  for (i=n;i>=1;i--) {
    if (i < n) {
      if (g) {
	for (j=l;j<=n;j++)
	  v[j][i]=(a[i][j]/a[i][l])/g;
	for (j=l;j<=n;j++) {
	  for (s=0.0,k=l;k<=n;k++) s += a[i][k]*v[k][j];
	  for (k=l;k<=n;k++) v[k][j] += s*v[k][i];
	}
      }
      for (j=l;j<=n;j++) v[i][j]=v[j][i]=0.0;
    }
    v[i][i]=1.0;
    g=rv1[i];
    l=i;
  }
  for (i=ss_IMIN(m,n);i>=1;i--) {
    l=i+1;
    g=w[i];
    for (j=l;j<=n;j++) a[i][j]=0.0;
    if (g) {
      g=1.0/g;
      
      for (j=l;j<=n;j++) {
	for (s=0.0,k=l;k<=m;k++) s += a[k][i]*a[k][j];
	f=(s/a[i][i])*g;
	for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
      }
      for (j=i;j<=m;j++) a[j][i] *= g;
    } else for (j=i;j<=m;j++) a[j][i]=0.0;
    ++a[i][i];
  }
  for (k=n;k>=1;k--) {
    for (its=1;its<=30;its++) {
      flag=1;
      for (l=k;l>=1;l--) {
	nm=l-1;
	if ((fabs(rv1[l])+anorm) == anorm) {
	  flag=0;
	  break;
	}
	if ((fabs(w[nm])+anorm) == anorm) break;
      }
      
      if (flag) {
	c=0.0;
	s=1.0;
	for (i=l;i<=k;i++) {
	  f=s*rv1[i];
	  rv1[i]=c*rv1[i];
	  if ((fabs(f)+anorm) == anorm) break;
	  g=w[i];
	  h=ss_pythag(f,g);
	  w[i]=h;
	  h=1.0/h;
	  c=g*h;
	  s = -f*h;
	  for (j=1;j<=m;j++) {
	    y=a[j][nm];
	    z=a[j][i];
	    a[j][nm]=y*c+z*s;
	    a[j][i]=z*c-y*s;
	  }
	}
      }
      z=w[k];
      if (l == k) {
	
	if (z < 0.0) {
	  w[k] = -z;
	  for (j=1;j<=n;j++) v[j][k] = -v[j][k];
	}
	break;
      }
      if (its == 30) ss_nrerror("no convergence in 30 svdcmp iterations");
      x=w[l];
      nm=k-1;
      y=w[nm];
      g=rv1[nm];
      h=rv1[k];
      f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
      g=ss_pythag(f,1.0);
      f=((x-z)*(x+z)+h*((y/(f+ss_SIGN(g,f)))-h))/x;
      c=s=1.0;
      for (j=l;j<=nm;j++) {
	i=j+1;
	g=rv1[i];
	y=w[i];
	h=s*g;

	g=c*g;
	z=ss_pythag(f,h);
	rv1[j]=z;
	c=f/z;
	s=h/z;
	f=x*c+g*s;
	g = g*c-x*s;
	h=y*s;
	y *= c;
	for (jj=1;jj<=n;jj++) {
	  x=v[jj][j];
	  z=v[jj][i];
	  v[jj][j]=x*c+z*s;
	  v[jj][i]=z*c-x*s;
	}
	z=ss_pythag(f,h);
	w[j]=z;
	if (z) {
	  z=1.0/z;
	  c=f*z;
	  s=h*z;
	}
	f=c*g+s*y;
	x=c*y-s*g;
	for (jj=1;jj<=m;jj++) {
	  
	  y=a[jj][j];
	  z=a[jj][i];
	  a[jj][j]=y*c+z*s;
	  a[jj][i]=z*c-y*s;
	}
      }
      rv1[l]=0.0;
      rv1[k]=f;
      w[k]=x;
    }
  }
  ss_free_dvector(rv1,1);
}

double ss_pythag(double a, double b)
{
    static double dsqrarg;
    double absa, absb;
    absa = fabs(a);
    absb = fabs(b);
    if (absa > absb) return absa * sqrt(1.0 + ss_DSQR(absb / absa));
    else return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + ss_DSQR(absa / absb)));
}

DMATRIX ss_xinvmat_svd(
		    DMATRIX mat,
		    double cond)	/* condition number */
{
    long ri, ci;
    double wmax, wmin;
    DVECTOR wvec;
    DVECTOR vvec;
    DMATRIX cpmat;
    DMATRIX transmat;
    DMATRIX vmat;
    DMATRIX invmat;
    DMATRIX invmat2 = NODATA;

    if (mat->row < mat->col) {
	cpmat = xdmzeros(mat->col, mat->col);
    } else {
	cpmat = xdmzeros(mat->row, mat->col);
    }
    for (ri = 0; ri < mat->row; ri++) {
	for (ci = 0; ci < mat->col; ci++) {
	    cpmat->data[ri][ci] = mat->data[ri][ci];
	}
    }
    wvec = xdvzeros(mat->col);
    vmat = xdmzeros(mat->col, mat->col);

    /* SVD */
    ss_svd(cpmat, wvec, vmat);
    /* search max-singular value */
    for (ri = 0, wmax = 0.0; ri < wvec->length; ri++) {
	if (wmax < wvec->data[ri]) {
	    wmax = wvec->data[ri];
	}
    }
    wmin = wmax * cond;
    /* calculate 1 / W */
    for (ri =0; ri < wvec->length; ri++) {
	if (wmin > wvec->data[ri] || wvec->data[ri] == 0.0) {
	    wvec->data[ri] = 0.0;
	} else {
	    wvec->data[ri] = 1.0 / wvec->data[ri];
	}
    }
    /* calculate inverse matrix */
    for (ri = 0; ri < vmat->row; ri++) {
	vvec = xdmextractrow(vmat, ri);
	dvoper(vvec, "*", wvec);
	dmcopyrow(vmat, ri, vvec);
	/* memory free */
	xdvfree(vvec);
    }
    transmat = ss_xtrans_mat(cpmat);
    invmat = ss_xmulti_mat(vmat, transmat);
    /* memory free */
    xdvfree(wvec);
    xdmfree(cpmat);
    xdmfree(transmat);
    xdmfree(vmat);

    if (mat->row < mat->col) {
	invmat2 = xdmalloc(mat->col, mat->row);
	for (ri = 0; ri < invmat2->row; ri++) {
	    for (ci = 0; ci < invmat2->col; ci++) {
		invmat2->data[ri][ci] = invmat->data[ri][ci];
	    }
	}
	/* memory free */
	xdmfree(invmat);

	return invmat2;
    } else {
	return invmat;
    }
}

void ss_svd(
	 DMATRIX a,	/* [m][n] */
	 DVECTOR w,	/* [n] */
	 DMATRIX v)	/* [n][n] */
{
    long ri, ci;
    DMATRIX ba;
    DVECTOR bw;
    DMATRIX bv;

    /* memory allocation */
    ba = xdmzeros(a->row + 1, a->col + 1);
    bw = xdvzeros(w->length + 1);
    bv = xdmzeros(v->row + 1, v->col + 1);

    /* bias */
    for (ri = 0; ri < a->row; ri++) {
	for (ci = 0; ci < a->col; ci++) {
	    ba->data[ri + 1][ci + 1] = a->data[ri][ci];
	}
    }
    for (ri = 0; ri < v->row; ri++) {
	bw->data[ri + 1] = w->data[ri];
	for (ci = 0; ci < v->col; ci++) {
	    bv->data[ri + 1][ci + 1] = v->data[ri][ci];
	}
    }
    /* SVD */
    ss_svdcmp(ba->data, a->row, a->col, bw->data, bv->data);
    /* remove bias */
    for (ri = 0; ri < a->row; ri++) {
	for (ci = 0; ci < a->col; ci++) {
	    a->data[ri][ci] = ba->data[ri + 1][ci + 1];
	}
    }
    for (ri = 0; ri < v->row; ri++) {
	w->data[ri] = bw->data[ri + 1];
	for (ci = 0; ci < v->col; ci++) {
	    v->data[ri][ci] = bv->data[ri + 1][ci + 1];
	}
    }

    /* memory free */
    xdmfree(ba);
    xdvfree(bw);
    xdmfree(bv);
}

DMATRIX ss_xdmclone(
		 DMATRIX mat)
{
    long ri;
    DVECTOR vec;
    DMATRIX clonemat;

    clonemat = xdmzeros(mat->row, mat->col);

    for (ri = 0; ri < mat->row; ri++) {
	vec = xdmextractrow(mat, ri);
	dmcopyrow(clonemat, ri, vec);
	xdvfree(vec);
    }

    return clonemat;
}

/* get transposition matrix */
DMATRIX ss_xtrans_mat(
		  DMATRIX mat)
{
    long ri, ci;
    DMATRIX trans;

    /* memory allocation */
    trans = xdmalloc(mat->col, mat->row);

    for (ri = 0; ri < trans->row; ri++) {
	for (ci = 0; ci < trans->col; ci++) {
	    trans->data[ri][ci] = mat->data[ci][ri];
	}
    }

    return trans;
}

DMATRIX ss_xmulti_mat(
		   DMATRIX mat1,
		   DMATRIX mat2)
{
    long ri, ci;
    DVECTOR mat1vec;
    DVECTOR mat2vec;
    DMATRIX multi;

    /* error check */
    if (mat1->col != mat2->row) {
	printmsg(stderr, "Can't multi matrixes\n");
	return NODATA;
    }
    /* memory allocation */
    multi = xdmalloc(mat1->row, mat2->col);

    for (ri = 0; ri < mat1->row; ri++) {
	for (ci = 0; ci < mat2->col; ci++) {
	    mat1vec = xdmextractrow(mat1, ri);
	    mat2vec = xdmextractcol(mat2, ci);
	    dvoper(mat1vec, "*", mat2vec);
	    xdvfree(mat2vec);
	    multi->data[ri][ci] = dvsum(mat1vec);
	    xdvfree(mat1vec);
	}
    }
    
    return multi;
}

/* change vector into matrix */
DMATRIX ss_xvec2matcol(
		    DVECTOR vec)
{
    DMATRIX mat;

    /* memory allocation */
    mat = xdmalloc(vec->length, 1);

    dmcopycol(mat, 0, vec);

    return mat;
}

/*
 *	read header of wav file
 */
char *xreadheader_wav(char *filename)
{
    char *header = NULL;
    FILE *fp;

    /* memory allocate */
    if ((header = (char *)malloc((int)44 * sizeof(char))) == NULL) {
	printmsg(stderr, "Read header: Memory allcation is failed\n");
	exit(0);
    }

    /* open file */
    if (NULL == (fp = fopen(filename, "rb"))) {
	printmsg(stderr, "can't open file: %s\n", filename);
	return NULL;
    }

    /* read data */
    fread(header, sizeof(char), (int)44, fp);

    /* close file */
    fclose(fp);

    return header;
}

/*
 *	write header of wav file
 */
void writessignal_wav(char *filename, SVECTOR vector, double fs)
{
    char riff[5] = "RIFF";
    char riffsize[4] = "";
    char wave[5] = "WAVE";
    char fmt[5] = "fmt ";
    char fmtsize[4] = "";
    unsigned short format = 1;
    unsigned short channel = 1;
    char sampling[4] = "";
    char bps[4] = "";
    char block = 2;
    char dummy1 = 0;
    char bit = 16;
    char dummy2 = 0;
    char data[5] = "data";
    char datasize[4] = "";
    unsigned long tmp;
    char *basicname;
    FILE *fp;

    fmtsize[3] = 0x00;	fmtsize[2] = 0x00;
    fmtsize[1] = 0x00;	fmtsize[0] = 0x10;
    
    tmp = (unsigned long)(vector->length * 2);
    datasize[3] = (char)(tmp >> 24);	datasize[2] = (char)(tmp >> 16);
    datasize[1] = (char)(tmp >> 8);	datasize[0] = (char)tmp;

    tmp += (unsigned long)36;
    riffsize[3] = (char)(tmp >> 24);	riffsize[2] = (char)(tmp >> 16);
    riffsize[1] = (char)(tmp >> 8);	riffsize[0] = (char)tmp;

    tmp = (unsigned long)fs;
    sampling[3] = (char)(tmp >> 24);	sampling[2] = (char)(tmp >> 16);
    sampling[1] = (char)(tmp >> 8);	sampling[0] = (char)tmp;

    tmp += tmp;
    bps[3] = (char)(tmp >> 24);	bps[2] = (char)(tmp >> 16);
    bps[1] = (char)(tmp >> 8);	bps[0] = (char)tmp;
    
    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdout")) {
	fp = stdout;
    } else {
	/* open file */
#ifndef WIN32
        check_dir(filename);
#endif
	if (NULL == (fp = fopen(filename, "wb"))) {
	    printmsg(stderr, "can't open file: %s\n", filename);
	    return;
	}
    }

    /* write header */
    fwrite(riff, sizeof(char), 4, fp);
    fwrite(riffsize, sizeof(char), 4, fp);
    fwrite(wave, sizeof(char), 4, fp);
    fwrite(fmt, sizeof(char), 4, fp);
    fwrite(fmtsize, sizeof(char), 4, fp);
    fwrite(&format, sizeof(unsigned short), 1, fp);
    fwrite(&channel, sizeof(unsigned short), 1, fp);
    fwrite(sampling, sizeof(char), 4, fp);
    fwrite(bps, sizeof(char), 4, fp);
    fwrite(&block, sizeof(char), 1, fp);
    fwrite(&dummy1, sizeof(char), 1, fp);
    fwrite(&bit, sizeof(char), 1, fp);
    fwrite(&dummy2, sizeof(char), 1, fp);
    fwrite(data, sizeof(char), 4, fp);
    fwrite(datasize, sizeof(char), 4, fp);
    
    /* write data */
    fwriteshort(vector->data, vector->length, 0, fp);

    /* close file */
    if (fp != stdout)
	fclose(fp);

    /* memory free */
    xfree(basicname);
    
    return;
}

void check_header(char *file, double *fs, XBOOL *float_flag, XBOOL msg_flag)
{
    char *header = NULL;
    
    // get header
    if ((header = xreadheader_wav(file)) == NULL) {
	printmsg(stderr, "Can't read header %s\n", file);
	exit(1);
    }
    // read header-information
    if ((strncmp(header, "RIFF", 4) != 0 ||
	 strncmp(header + 8, "WAVE", 4) != 0) ||
	(strncmp(header + 12, "fmt", 3) != 0 ||
	 strncmp(header + 36, "data", 4) != 0)) {
	printmsg(stderr, "no wav file: %s\n", file);
	exit(1);
    } else {
	if (fs != NULL) {
	    *fs = (double)(((((header[27] << 24) & 0xff000000) |
			     ((header[26] << 16) & 0xff0000))) |
			   ((((header[25] << 8) & 0xff00) |
			     ((header[24]) & 0xff))));
	    if (msg_flag == XTRUE)
		printmsg(stderr, "Sampling frequency %5.0f [Hz]\n", *fs);
	}
    }
    if (header[34] == 16) {
	if (msg_flag == XTRUE) printmsg(stderr, "16bit short wave\n");
	*float_flag = XFALSE;
    } else if (header[34] == 32) {
	if (msg_flag == XTRUE) printmsg(stderr, "32bit float wave\n");
	*float_flag = XTRUE;
    } else {
	printmsg(stderr, "no wav file: %s\n", file);
	printmsg(stderr, "Please use this type: signed 16 bit or float 32 bit\n");
	exit(1);
    }
    xfree(header);

    return;
}


DVECTOR xread_wavfile(char *file, double *fs, XBOOL msg_flag)
{
    long headlen = 44;
    SVECTOR xs = NODATA;
    FVECTOR xf = NODATA;
    DVECTOR xd = NODATA;
    XBOOL float_flag = XFALSE;

    /* read header */
    check_header(file, fs, &float_flag, msg_flag);
    // read waveform
    if (float_flag == XFALSE) {
	/* read short wave data */
	if ((xs = xreadssignal(file, headlen, 0)) == NODATA) {
	    exit(1);
	} else {
	    xd = xsvtod(xs);	xsvfree(xs);
	}
    } else {
	/* read float wave data */
	if ((xf = xreadfsignal(file, headlen, 0)) == NODATA) {
	    exit(1);
	} else {
	    xd = xfvtod(xf);	xfvfree(xf);
	    dvscoper(xd, "*", 32000.0);
	}
    }
    if (msg_flag == XTRUE) printmsg(stderr, "read wave: %s\n", file);

    return xd;
}
