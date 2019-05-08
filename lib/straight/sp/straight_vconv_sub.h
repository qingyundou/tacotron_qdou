/* straight_vconv_sub.h
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __STRAIGHT_VCONV_SUB_H
#define __STRAIGHT_VCONV_SUB_H


extern DVECTOR xget_wave2powvec(DVECTOR xd, double fs, double frame,
				double shift, long len, XBOOL log_flag);
extern DVECTOR xget_spg2powvec(DMATRIX n2sgram, XBOOL log_flag);
extern double spvec2pow(DVECTOR vec, XBOOL db_flag);
extern DVECTOR xspg2pow_norm(DMATRIX spg);
extern double *dalloc(int cc);
extern void fillz(double *ptr, int nitem);
extern void movem(register double *a, register double *b, int nitem);
extern void freqt(double *c1, long m1, double *c2, long m2, double a);
extern DVECTOR xcep2mcep(DVECTOR cep, long order, long fftl, XBOOL power_flag,
			 XBOOL inv_flag);
extern DVECTOR xcep2mpmcep(DVECTOR cep, long order, long fftl,
                           XBOOL power_flag, XBOOL inv_flag, double alpha);
extern DVECTOR xget_spw2cep(DVECTOR spw, long order, XBOOL power_flag);
extern DVECTOR xget_cep2spw(DVECTOR cep, long fftl);
extern DVECTOR xget_vec2cep(DVECTOR vec, long order, XBOOL power_flag);
extern DVECTOR xget_cep2vec(DVECTOR cep, long fftl);
extern DMATRIX xget_spg2cepg(DMATRIX spg, long order, XBOOL power_flag);
extern DMATRIX xget_cepg2spg(DMATRIX cepg, long fftl);
extern DMATRIX xcepg2mcepg(DMATRIX cepg, long order, long fftl,
                           XBOOL power_flag, XBOOL inv_flag);
extern DMATRIX xcepg2mpmcepg(DMATRIX cepg, long order, long fftl,
                             XBOOL power_flag, XBOOL inv_flag, double alpha);
extern DMATRIX xget_fftmat(DMATRIX mat, long fftl, long order, XBOOL inv_flag);



#endif /* __STRAIGHT_VCONV_SUB_H */
