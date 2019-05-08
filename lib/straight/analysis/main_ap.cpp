/*
 *    main.c : main function of straight
 *
 *        coded by H.Banno          1996/12/25
 *        modified by T. Toda       2001/2/12
 *            V30k18 (matlab)
 *
 *        modified by J. Yamagishi  2010/3/26
 *
 *
 *    STRAIGHT Analysis applied to concatenative TTS
 *        coded by T. Toda
 *
 *    Tomoki Toda (tomoki.toda@atr.co.jp)
 *            From Mar. 2001 to Sep. 2003
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fileio.h"
#include "option.h"
#include "voperate.h"

#include "straight_sub.h"
#include "straight_body_sub.h"
#include "straight_vconv_sub.h"

#include "straight_body_c03.h"

typedef struct CONDITION_STRUCT {
  double fs;          /* Sampling frequency [Hz] */
  double shiftm;      /* Frame shift (ms) */
  double f0shiftm;    /* Frame shift (ms) */
  long fftl;          /* FFT length */
  long apord;         /* IFFT Aperiodic order */
  char *f0file;       /* F0 text file */
  char *cf0file;      /* candidate F0 text file */
  XBOOL wav_flag;     /* wav file */
  XBOOL bndap_flag;   /* wav file */
  XBOOL float_flag;   /* float */
  XBOOL msg_flag;     /* print message */
  XBOOL help_flag;
} CONDITION;

CONDITION cond = {16000.0, 5.0, 5.0, 512, 257, NULL, NULL,
                  XTRUE, XFALSE, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
  {"[inputfile]", NULL},
  {"[outputfile]", NULL},
};

#define NUM_OPTION 12
OPTION option_struct[] = {
  {"-f", NULL, "sampling frequency [Hz]", "samp_freq", 
   NULL, TYPE_DOUBLE, &cond.fs, XFALSE},
  {"-shift", NULL, "frame shift [ms]", "shift", 
   NULL, TYPE_DOUBLE, &cond.shiftm, XFALSE},
  {"-f0shift", NULL, "F0 frame shift [ms]", "f0shift", 
   NULL, TYPE_DOUBLE, &cond.f0shiftm, XFALSE},
  {"-fftl", NULL, "fft length", "fft_length", 
   NULL, TYPE_LONG, &cond.fftl, XFALSE},
  {"-apord", NULL, "IFFT aperiodic energy order", "apord", 
   NULL, TYPE_LONG, &cond.apord, XFALSE},
  {"-f0file", NULL, "F0 filename", "f0file", 
   NULL, TYPE_STRING, &cond.f0file, XFALSE},
  {"-cf0file", NULL, "candidate F0 filename", "cf0file", 
   NULL, TYPE_STRING, &cond.cf0file, XFALSE},
  {"-raw", NULL, "input raw file (16bit short)", NULL, 
   NULL, TYPE_BOOLEAN, &cond.wav_flag, XFALSE},
  {"-bndap", NULL, "critical band aperiodic energy", NULL, 
   NULL, TYPE_BOOLEAN, &cond.bndap_flag, XFALSE},
  {"-float", NULL, "output float", NULL, 
   NULL, TYPE_BOOLEAN, &cond.float_flag, XFALSE},
  {"-nmsg", NULL, "no message", NULL, 
   NULL, TYPE_BOOLEAN, &cond.msg_flag, XFALSE},
  {"-help", "-h", "display this message", NULL, 
   NULL, TYPE_BOOLEAN, &cond.help_flag, XFALSE},
};

OPTIONS options_struct = {
  NULL, 1, NUM_OPTION, option_struct, NUM_ARGFILE, argfile_struct,
};

/* main */
int main(int argc, char *argv[])
{
  int i, fc;
  long k, l, b, bs, be;
  SVECTOR x = NODATA;
  DVECTOR xd = NODATA;
  DVECTOR f0l = NODATA;
  DVECTOR if0l = NODATA;
  DMATRIX dapv = NODATA;
  DMATRIX idapv = NODATA;
  DMATRIX avdapv = NODATA;
  
  /* get program name */
  options_struct.progname = xgetbasicname(argv[0]);

    /* get option */
  for (i = 1, fc = 0; i < argc; i++)
    if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
      getargfile(argv[i], &fc, &options_struct);
  
  /* display message */
  if (cond.help_flag == XTRUE)
    printhelp(options_struct, "Speech Transformation and Representation using Adaptive Interpolaiton of weiGHTed spectrogram");

  if (fc != options_struct.num_file)
    printerr(options_struct, "not enough files");
  
  if (cond.wav_flag == XTRUE) {

    xd = xread_wavfile(options_struct.file[0].name, &cond.fs, cond.msg_flag);

  } else {

    /* read wave data */
    if ((x = xreadssignal(options_struct.file[0].name, 0, 0)) == NODATA) {
      fprintf(stderr, "Can't read wave data\n");
      exit(1);
    } else {
      if (cond.msg_flag == XTRUE)
        fprintf(stderr, "read wave: %s\n", options_struct.file[0].name);
      xd = xsvtod(x);
      xsvfree(x);
    }
  }

  /* SOURCE INFORMATION */
  if (strnone(cond.f0file) && strnone(cond.cf0file)) {

    fprintf(stderr, "Should be use F0 file, -f0file or -cf0file option\n");
    exit(1);

  } else if (!strnone(cond.f0file)) {

    /* read f0 file */
    if ((f0l = xdvreadcol_txt(cond.f0file, 0)) == NODATA) {
      fprintf(stderr, "straight: can't read F0 file\n");
      exit(1);
    } else {
      if (cond.msg_flag == XTRUE)
        fprintf(stderr, "read F0: %s\n", cond.f0file);
    }

  } else if (!strnone(cond.cf0file)) {

    /* read f0 file */
    if ((f0l = xdvread_cf0file(cond.cf0file, &cond.f0shiftm)) == NODATA) {
      fprintf(stderr, "straight: can't read candidate F0 file\n");
      exit(1);
    } else {
      if (cond.msg_flag == XTRUE)
        fprintf(stderr, "read F0: %s\n", cond.cf0file);
    }

  }

  // interpolation
  if (cond.msg_flag == XTRUE)
    fprintf(stderr, "F0shift %f[ms] => %f[ms]\n", cond.f0shiftm, cond.shiftm);
  if0l = xget_interp_f0(f0l, cond.f0shiftm, cond.shiftm);
  // memory free
  xdvfree(f0l);

  /* MULTI BAND ANALYSIS */
  if (cond.msg_flag == XTRUE)
    fprintf(stderr, "     === MBE type analysis ===\n");
  /* calculate aperiodic energy */
  if ((dapv = aperiodicpart4(xd, if0l, cond.fs, cond.shiftm, 5.0, cond.fftl / 2 + 1, cond.msg_flag)) 
      == NODATA) {
    fprintf(stderr, "straight: MBE type analysis failed\n");
    exit(1);
  }
  
  if (cond.msg_flag == XTRUE)
    fprintf(stderr, "         relative aperiodic energy\n");
  
  if (cond.bndap_flag == XTRUE) {

    /*
      Calc number of critical bands and their edge frequency from sampling frequency
                                 Added by J. Yamagishi (28 March 2010)
      Reference 
      http://en.wikipedia.org/wiki/Bark_scale
      
      Zwicker, E. (1961), "Subdivision of the audible frequency range
      into critical bands," The Journal of the Acoustical Society of
      America, 33, Feb., 1961.
      
      H. Traunmuller (1990) "Analytical expressions for the tonotopic
      sensory scale" J. Acoust. Soc. Am. 88: 97-100.      
    */
    
    int  nq, numbands;
    float fbark, startf, endf;
     
    // Calc the number of critical bands required for sampling frequency
    nq = cond.fs / 2;     
    fbark = 26.81 * nq / (1960 + nq ) - 0.53;  
    if(fbark<2)
      fbark += 0.15*(2-fbark);  
    if(fbark>20.1)
      fbark +=  0.22*(fbark-20.1);
    numbands = (int) (fbark + 0.5); 

    avdapv = xdmalloc(dapv->row, numbands);

    for (b = 0; b < numbands; b++) {
      startf = 1960 / (26.81 / (b  + 0.53) - 1);
      startf = round (startf / 100 ) * 100;      
      endf = 1960 / (26.81 / (b + 1 + 0.53) - 1);
      endf = round (endf / 100 ) * 100;

      if (startf < 20.0) 
        startf = 20;  // human hearing (20 Hz - 20kHz)

      // deltaf = sampling frequency / fftl 
      bs = startf * cond.fftl / cond.fs;
      if (endf < nq) 
        be = endf * cond.fftl / cond.fs;       
      else
        be = cond.fftl / 2 + 1;

      for (k = 0; k < dapv->row; k++) {
        for (l = bs, avdapv->data[k][b] = 0.0; l < be; l++)
          avdapv->data[k][b] += dapv->data[k][l];
        avdapv->data[k][b] /= (double)(be - bs);
      }
    }

    /* write aperiodic energy file */
    if (cond.float_flag == XFALSE) {
      writedmatrix(options_struct.file[1].name, avdapv, 0);
    } else {
      writed2fmatrix(options_struct.file[1].name, avdapv, 0);
    }
    if (cond.msg_flag == XTRUE)
      fprintf(stderr, "write critical band aperiodic energy [%ld][%ld]: %s\n",
              avdapv->row, avdapv->col, options_struct.file[1].name);
    /* memory free */
    xdmfree(avdapv);
  } else {
    if (cond.apord < cond.fftl / 2 + 1) {
      idapv = xget_fftmat(dapv, cond.fftl, cond.apord, XTRUE);
      /* write aperiodic energy file */
      if (cond.float_flag == XFALSE) {
        writedmatrix(options_struct.file[1].name, idapv, 0);
      } else {
        writed2fmatrix(options_struct.file[1].name, idapv, 0);
      }
      if (cond.msg_flag == XTRUE)
        fprintf(stderr, "write IFFT aperiodic energy [%ld][%ld]: %s\n",
                idapv->row, idapv->col, options_struct.file[1].name);
      /* memory free */
      xdmfree(idapv);
    } else {
      /* write aperiodic energy file */
      if (cond.float_flag == XFALSE) {
        writedmatrix(options_struct.file[1].name, dapv, 0);
      } else {
        writed2fmatrix(options_struct.file[1].name, dapv, 0);
      }
      if (cond.msg_flag == XTRUE)
        fprintf(stderr, "write aperiodic energy [%ld][%ld]: %s\n",
                dapv->row, dapv->col, options_struct.file[1].name);
    }
  }

  /* memory free */
  xdvfree(xd);
  xdmfree(dapv);
  
  return 0;
}
