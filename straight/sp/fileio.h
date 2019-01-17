/* fileio.h
 *	coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __FILEIO_H
#define __FILEIO_H

#include "vector.h"
#include "matrix.h"

extern long getfilesize(char *filename, int headlen);

extern void swapshort(short *data, long length);
extern void swaplong(long *data, long length);
extern void swapfloat(float *data, long length);
extern void swapdouble(double *data, long length);

extern void freadshort(short *data, long length, int swap, FILE *fp);
extern void freadlong(long *data, long length, int swap, FILE *fp);
extern void freadfloat(float *data, long length, int swap, FILE *fp);
extern void freaddouble(double *data, long length, int swap, FILE *fp);
extern void freadshorttod(double *data, long length, int swap, FILE *fp);
extern void fwriteshort(short *data, long length, int swap, FILE *fp);
extern void fwritelong(long *data, long length, int swap, FILE *fp);
extern void fwritefloat(float *data, long length, int swap, FILE *fp);
extern void fwritedouble(double *data, long length, int swap, FILE *fp);
extern void fwritedoubletos(double *data, long length, int swap, FILE *fp);

extern SVECTOR xreadssignal(char *filename, int headlen, int swap);
extern FVECTOR xreadfsignal(char *filename, int headlen, int swap);
extern DVECTOR xdvreadssignal(char *filename, int headlen, int swap);
extern DVECTOR xreaddsignal(char *filename, int headlen, int swap);
extern DVECTOR xreadf2dsignal(char *filename, int headlen, int swap);
extern void writessignal(char *filename, SVECTOR vector, int swap);
extern void dvwritessignal(char *filename, DVECTOR vector, int swap);
extern void writedsignal(char *filename, DVECTOR vector, int swap);
extern void writed2fsignal(char *filename, DVECTOR vector, int swap);

extern LMATRIX xreadlmatrix(char *filename, long ncol, int swap);
extern DMATRIX xreaddmatrix(char *filename, long ncol, int swap);
extern DMATRIX xreadf2dmatrix(char *filename, long ncol, int swap);
extern void writelmatrix(char *filename, LMATRIX mat, int swap);
extern void writedmatrix(char *filename, DMATRIX mat, int swap);
extern void writed2fmatrix(char *filename, DMATRIX mat, int swap);

extern long getfilesize_txt(char *filename);
extern int readdvector_txt(char *filename, DVECTOR vector);
extern DVECTOR xreaddvector_txt(char *filename);
extern int writedvector_txt(char *filename, DVECTOR vector);

extern int getnumrow_txt(char *filename);
extern int getnumcol_txt(char *filename);
extern int fgetcol_txt(char *buf, int col, FILE *fp);
extern int sgetcol(char *buf, int col, char *line);
extern int fgetline(char *buf, FILE *fp);
extern int getline(char *buf);
extern char *gets0(char *buf, int size);
extern int sscanf_setup(char *line, char *name, char *value);
extern int dvreadcol_txt(char *filename, int col, DVECTOR vector);
extern DVECTOR xdvreadcol_txt(char *filename, int col);

#define fgetnumrow getnumrow_txt
#define fgetnumcol getnumcol_txt
#define fgetcol fgetcol_txt

extern void svdump(SVECTOR vec);
extern void lvdump(LVECTOR vec);
extern void fvdump(FVECTOR vec);
extern void dvdump(DVECTOR vec);

extern void svfdump(SVECTOR vec, FILE *fp);
extern void lvfdump(LVECTOR vec, FILE *fp);
extern void fvfdump(FVECTOR vec, FILE *fp);
extern void dvfdump(DVECTOR vec, FILE *fp);

extern void lmfdump(LMATRIX mat, FILE *fp);
extern void dmfdump(DMATRIX mat, FILE *fp);

#ifdef VARARGS
extern void dvnfdump();
#else
extern void dvnfdump(FILE *fp, DVECTOR vec, ...);
#endif

#define xreadsvector(filename, swap) xreadssignal((filename), 0, (swap))
#define writesvector writessignal

#define xreaddvector(filename, swap) xreaddsignal((filename), 0, (swap))
#define writedvector writedsignal

#define read_dvector_txt readdvector_txt
#define write_dvector_txt writedvector_txt

#define getsiglen(filename, headlen, type) (getfilesize(filename, headlen) / (long)sizeof(type))
#define getisiglen(filename, headlen) (getsiglen(filename, headlen, int))
#define getssiglen(filename, headlen) (getsiglen(filename, headlen, short))
#define getlsiglen(filename, headlen) (getsiglen(filename, headlen, long))
#define getfsiglen(filename, headlen) (getsiglen(filename, headlen, float))
#define getdsiglen(filename, headlen) (getsiglen(filename, headlen, double))

#ifndef WIN32
extern void check_dir(char *file);
#endif


#endif /* __FILEIO_H */
