/* straight_synth_tb06_unit.h
 *
 *    Tomoki Toda (tomoki.toda@atr.co.jp)
 *            From Mar. 2001 to Sep. 2003
 */

#ifndef __STRAIGHT_SYNTH_TB06_UNIT_H
#define __STRAIGHT_SYNTH_TB06_UNIT_H

/* STRAIGHT Synthesis Using Parameters Generated from HMM */
extern DVECTOR straight_synth_tb06(DMATRIX n2sgram, DVECTOR f0l,
                                   DVECTOR f0var, double fs, double shiftm,
                                   double pconv, double fconv, double sconv,
                                   double gdbw, double delsp, double cornf,
                                   double delfrac, XBOOL fr_flag,
                                   XBOOL zp_flag, XBOOL rp_flag,
                                   XBOOL df_flag);

extern DVECTOR straight_synth_tb06ca(DMATRIX n2sgram, DVECTOR f0l, double fs,
                                     double shiftm, double sigp, double pconv,
                                     double fconv, double sconv, double gdbw,
                                     double delsp, double cornf,
                                     double delfrac, DMATRIX ap, DVECTOR imap,
                                     XBOOL bap_flag,
                                     XBOOL fr_flag, XBOOL zp_flag,
                                     XBOOL rp_flag, XBOOL df_flag);

extern DMATRIX xread_dfcep2spg(char *file, long dim, long fftl, XBOOL mel_flag,
                               XBOOL float_flag, XBOOL chpow_flag,
                               double frame, double fs, double alpha);

#endif /* __STRAIGHT_SYNTH_TB06_UNIT_H */
