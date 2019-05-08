/* straight_body_sub.h
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __STRAIGHT_SUB_H
#define __STRAIGHT_SUB_H
   
/*static double dsqrarg;*/
#define ss_DSQR(a) ((dsqrarg=(a)) == 0.0 ? 0.0 : dsqrarg*dsqrarg)
   
/*static double dmaxarg1,dmaxarg2;*/
#define ss_DMAX(a,b) (dmaxarg1=(a),dmaxarg2=(b),(dmaxarg1) > (dmaxarg2) ?\
        (dmaxarg1) : (dmaxarg2))
   
/*static int iminarg1,iminarg2;*/
#define ss_IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
        (iminarg1) : (iminarg2))
   
#define ss_SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
   
extern void ss_nrerror(char error_text[]);
extern double *ss_dvector(long nl, long nh);

extern void ss_free_dvector(double *v, long nl);
extern void ss_svdcmp(double **a, long m, long n, double w[], double **v);
extern double ss_pythag(double a, double b);
extern DMATRIX ss_xinvmat_svd(DMATRIX mat, double cond);
extern void ss_svd(DMATRIX a, DVECTOR w, DMATRIX v);
extern DMATRIX ss_xdmclone(DMATRIX mat);
extern DMATRIX ss_xtrans_mat(DMATRIX mat);
extern DMATRIX ss_xmulti_mat(DMATRIX mat1, DMATRIX mat2);
extern DMATRIX ss_xvec2matcol(DVECTOR vec);
extern char *xreadheader_wav(char *filename);
extern void writessignal_wav(char *filename, SVECTOR vector, double fs);
extern void check_header(char *file, double *fs, XBOOL *float_flag,
			XBOOL msg_flag);
extern DVECTOR xread_wavfile(char *file, double *fs, XBOOL msg_flag);


#endif /* _STRAIGHT_SUB_H_ */
