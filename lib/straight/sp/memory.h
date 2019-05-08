/* memory.h
 *	coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __MEMORY_H
#define __MEMORY_H

extern char *safe_malloc(unsigned int nbytes);
extern char *safe_realloc(char *p, unsigned int nbytes);

extern int **imatalloc(int row, int col);
extern void imatfree(int **mat, int row);

extern short **smatalloc(int row, int col);
extern void smatfree(short **mat, int row);

extern long **lmatalloc(int row, int col);
extern void lmatfree(long **mat, int row);

extern float **fmatalloc(int row, int col);
extern void fmatfree(float **mat, int row);

extern double **dmatalloc(int row, int col);
extern void dmatfree(double **mat, int row);

extern char *strclone(char *string);

#define xalloc(n, type) (type *)safe_malloc((unsigned)(n)*sizeof(type))
#define xrealloc(p, n, type) (type *)safe_realloc((char *)(p),(unsigned)(n)*sizeof(type))
#define xfree(p) {free((char *)(p));(p)=NULL;}

#define arrcpy(p1, p2, n, type) memmove((char *)(p1),(char *)(p2),(unsigned)(n)*sizeof(type))

#define strrepl(s1, s2) {if ((s1) != NULL) xfree(s1); (s1) = (((s2) != NULL) ? strclone(s2) : NULL);}
#define strreplace strrepl

#define xsmatalloc(row, col) smatalloc((int)(row), (int)(col))
#define xlmatalloc(row, col) lmatalloc((int)(row), (int)(col))
#define xfmatalloc(row, col) fmatalloc((int)(row), (int)(col))
#define xdmatalloc(row, col) dmatalloc((int)(row), (int)(col))

#define xsmatfree(x, row) smatfree(x, (int)(row))
#define xlmatfree(x, row) lmatfree(x, (int)(row))
#define xfmatfree(x, row) fmatfree(x, (int)(row))
#define xdmatfree(x, row) dmatfree(x, (int)(row))

#endif /* __MEMORY_H */
