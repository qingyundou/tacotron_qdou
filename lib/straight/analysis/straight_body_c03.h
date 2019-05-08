/* straight_body_c03.h
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __STRAIGHT_BODY_C03_H
#define __STRAIGHT_BODY_C03_H

/* for V30k18 */
extern DMATRIX straight_body_c03(DVECTOR x, DVECTOR f0l, double fs,
				 double framem, double shiftm, long fftl,
				 double eta, double pc, XBOOL fast_flag,
				 XBOOL msg_flag);

#endif /* __STRAIGHT_BODY_C03_H */
