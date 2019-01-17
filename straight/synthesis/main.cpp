/*
 *    main.c : main function of straight
 *
 *        coded by H.Banno     1996/12/25
 *        coded by H.Banno     1996/12/25
 *
 *        modified by T. Toda    2001/2/12
 *            V30k18 (matlab)
 *
 *        modified by J. Yamagishi  from 2007 to 2010
 *
 *    STRAIGHT Synthesis applied to HMM Speech Synthesis
 *        coded by T. Toda
 *
 *    Tomoki Toda (tomoki.toda@atr.co.jp)
 *            From Mar. 2001 to Sep. 2003
 *    
 *    Junichi Yamagishi (jyamagis@inf.ed.ac.uk)
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
#include "straight_synth_sub.h"

#include "straight_synth_tb06.h"

typedef struct CONDITION_STRUCT {
    double fs;        /* Sampling frequency [Hz] */
    double shiftm;    /* Frame shift (ms) */
    long fftl;        /* FFT length */
    long dim;         /* dimension of cepstrum */
    double sigp;
    double delsp;     /* Standard deviation of group delay (ms) */
    double gdbw;      /* Finest resolution in group delay (Hz) */
    double cornf;     /* Lower corner frequency for random phase (Hz) */
    double delfrac;   /* Ratio of standard deviation of group delay */
    double pc;
    double fc;
    double sc;
    char *apfile;     /* Aperiodic energy */
    XBOOL df_flag;    /* Proportional group delay */
    XBOOL mel_flag;   /* Use Mel cepstrum */
    double alpha;     /* Frequency warping factor for all-pass filter used for mel-cepstrum */
    XBOOL spec_flag;  /* Use spectrum (disable mel_flag) */
    XBOOL bap_flag;   /* Band-limited aperiodicity flag */
    XBOOL float_flag; /* float format */
    XBOOL wav_flag;   /* wav file */
    XBOOL msg_flag;   /* print message */
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {16000.0, 5.0, 1024, 24, 0.0, 0.5, 70.0, 3500.0, 0.2,
                  1.0, 1.0, 1.0, NULL, XFALSE, XFALSE, 0.42, XFALSE, XFALSE, XFALSE,
                  XTRUE, XTRUE, XFALSE};

#define NUM_ARGFILE 3
ARGFILE argfile_struct[] = {
    {"[f0file]", NULL},
    {"[cepfile]", NULL},
    {"[outputfile]", NULL},
};

#define NUM_OPTION 20
OPTION option_struct[] = {
    {"-f", NULL, "sampling frequency [Hz]", "samp_freq", 
     NULL, TYPE_DOUBLE, &cond.fs, XFALSE},
    {"-shift", NULL, "frame shift [ms]", "shift", 
     NULL, TYPE_DOUBLE, &cond.shiftm, XFALSE},
    {"-fftl", NULL, "fft length", "fft_length", 
     NULL, TYPE_LONG, &cond.fftl, XFALSE},
    {"-order", NULL, "cepstrum order", "order", 
     NULL, TYPE_LONG, &cond.dim, XFALSE},
    {"-sigp", NULL, "sigmoid parameter", "sigp", 
     NULL, TYPE_DOUBLE, &cond.sigp, XFALSE},
    {"-sd", NULL, "standard deviation of group delay", "sd", 
     NULL, TYPE_DOUBLE, &cond.delsp, XFALSE},
    {"-bw", NULL, "band width of group delay ", "bw", 
     NULL, TYPE_DOUBLE, &cond.gdbw, XFALSE},
    {"-cornf", NULL, "corner frequency for random phase", "cornf", 
     NULL, TYPE_DOUBLE, &cond.cornf, XFALSE},
    {"-delfrac", NULL, "ratio of stand. dev. of group delay",
     "delfrac", NULL, TYPE_DOUBLE, &cond.delfrac, XFALSE},
    {"-pc", NULL, "pitch scale conversion",
     "pc", NULL, TYPE_DOUBLE, &cond.pc, XFALSE},
    {"-fc", NULL, "frequency scale conversion",
     "fc", NULL, TYPE_DOUBLE, &cond.fc, XFALSE},
    {"-sc", NULL, "time scale conversion",
     "sc", NULL, TYPE_DOUBLE, &cond.sc, XFALSE},
    {"-apfile", NULL, "aperiodic energy file", "apfile",
     NULL, TYPE_STRING, &cond.apfile, XFALSE},
    {"-df", NULL, "using proportional group delay", NULL, 
     NULL, TYPE_BOOLEAN, &cond.df_flag, XFALSE},
    {"-mel", NULL, "Use mel cepstrum", NULL, 
     NULL, TYPE_BOOLEAN, &cond.mel_flag, XFALSE},
    {"-alpha", NULL, "warping factor", "alpha",
     NULL, TYPE_DOUBLE, &cond.alpha, XFALSE},
    {"-spec", NULL, "Use spectrum", NULL, 
     NULL, TYPE_BOOLEAN, &cond.spec_flag, XFALSE},
    {"-bap", NULL, "Band-limited aperiodicity", NULL, 
     NULL, TYPE_BOOLEAN, &cond.bap_flag, XFALSE},
    {"-float", NULL, "input float cepstrogram", NULL, 
     NULL, TYPE_BOOLEAN, &cond.float_flag, XFALSE},
    {"-raw", NULL, "output raw file (16 bit short)", NULL, 
     NULL, TYPE_BOOLEAN, &cond.wav_flag, XFALSE},
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
    long k;
    SVECTOR xo = NODATA;
    DVECTOR sy = NODATA;
    DVECTORS f0info = NODATA;
    DMATRIX spg = NODATA;
    /* DMATRIX dapv = NODATA; */
    DMATRIX ap = NODATA;

    /* get program name */
    options_struct.progname = xgetbasicname(argv[0]);
    
    /* get option */
    for (i = 1, fc = 0; i < argc; i++) {
      if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
        getargfile(argv[i], &fc, &options_struct);
    }

    /* display message */
    if (cond.help_flag == XTRUE) 
      printhelp(options_struct, "Speech Transformation and Representation using Adaptive Interpolaiton of weiGHTed spectrogram");

    if (fc != options_struct.num_file)
      printerr(options_struct, "not enough files");

    if((cond.mel_flag == XTRUE) && (cond.spec_flag == XTRUE)){
      fprintf(stderr, "straight: can't use mel flag and spec flag at the same time. Please choose one of them.\n");
      exit(1);
    }
    
    /* read f0 file */
    f0info = xdvsalloc(2);
    if ((f0info->vector[0] =
         xreaddvector_txt(options_struct.file[0].name)) == NODATA) {
      fprintf(stderr, "straight: can't read F0 file\n");
      exit(1);
    }
    if (cond.msg_flag == XTRUE)
      fprintf(stderr, "read F0[%ld]\n", f0info->vector[0]->length);
    
    if(cond.spec_flag == XTRUE){
      /* read spectrum directly */
        if (cond.float_flag == XFALSE) {    
          // double binary
          if ((spg = xreaddmatrix(options_struct.file[1].name, cond.fftl / 2 + 1, 0)) == NODATA) {
            fprintf(stderr,
                    "straight: can't read spectrum file\n");
            exit(1);
          }
        } else { 
          // float binary
          if ((spg=xreadf2dmatrix(options_struct.file[1].name, cond.fftl / 2 + 1, 0)) == NODATA) {
            fprintf(stderr,
                    "straight: can't read spectrum file\n");
            exit(1);
          }
        }
        fprintf(stderr, "read spectrum[%ld][%ld]\n", spg->row, spg->col);
    }else{
      /* read mel-cepstrum or cepstrum file and convert to spectrum */
      if ((spg = xread_dfcep2spg(options_struct.file[1].name, cond.dim + 1, cond.fftl, cond.mel_flag, cond.float_flag,
                                 XFALSE, 20.0,  cond.fs, cond.alpha)) == NODATA) {
        fprintf(stderr, "straight: can't read cepstrogram file\n");
        exit(1);
      }else{
        if (cond.msg_flag == XTRUE) {
          if(cond.mel_flag==XTRUE)
            fprintf(stderr, "read Mel Cepstrogram [%f]-> ",cond.alpha);
        else
          fprintf(stderr, "read Cepstrogram -> ");
        fprintf(stderr, " Spectrogram[%ld][%ld]\n", spg->row, spg->col);
        }
      }
    }
    
    // if (0) {
    //  writedmatrix(options_struct.file[2].name, spg, 0);
    //  exit(1);
    //}
    
    // read aperiodic energy file
    if (!strnone(cond.apfile)) {
      /* aperiodicity files --> mixed exciation */
      if (cond.bap_flag == XTRUE) {   
        // band-limited aperiodicity case         
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
        float fbark;
        
        // Calc the number of critical bands required for sampling frequency
        nq = cond.fs / 2;     
        fbark = 26.81 * nq / (1960 + nq ) - 0.53;  
        if(fbark<2)
          fbark += 0.15*(2-fbark);  
        if(fbark>20.1)
          fbark +=  0.22*(fbark-20.1);
        numbands = (int) (fbark + 0.5); 
        
        if (cond.float_flag == XFALSE) {  
          // double
          if ( ((ap = xreaddmatrix(cond.apfile, numbands, 0))==NODATA) || ( ap->row != f0info->vector[0]->length ) ) {
              // do the classic mode (5 bands)
              if ((ap = xreaddmatrix(cond.apfile, 5, 0))
                  == NODATA) {
              fprintf(stderr,
                    "straight: can't read aperiodic energy file\n");
              exit(1);
            }
          }
        } else {    
          // float
          if ((ap = xreadf2dmatrix(cond.apfile, numbands, 0))
              == NODATA) {
            fprintf(stderr,
                    "straight: can't read aperiodic energy file\n");
            exit(1);
          }
        }
        /* if frame-shift periods of F0 and aperiodicty are different, do interpolation */
        /* 
           if ((ap = aperiodiccomp(dapv, cond.shiftm, f0info->vector[0], cond.shiftm)) == NODATA) { 
           fprintf(stderr,
           "straight: aperiodic energy interpolation failed\n");
           exit(1);
           }*/
        /* memory free */
        /* xdmfree(dapv); */
        
      }else{
        // Aperiodicity case (fft/2 + 1)
        if (cond.float_flag == XFALSE) {    
          // double
          if ((ap = xreaddmatrix(cond.apfile, cond.fftl / 2 + 1, 0)) == NODATA) {
            fprintf(stderr,
                    "straight: can't read aperiodic energy file\n");
            exit(1);
          }
        } else {
          // float
          if ((ap=xreadf2dmatrix(cond.apfile, cond.fftl / 2 + 1, 0)) == NODATA) {
            fprintf(stderr,
                    "straight: can't read aperiodic energy file\n");
            exit(1);
          }
        }
      }

      if (cond.msg_flag == XTRUE)
        fprintf(stderr, "read aperiodic energy[%ld][%ld]\n", ap->row, ap->col);
      /* synthesize speech */
      if (cond.msg_flag == XTRUE)
        fprintf(stderr,
                "    === STRAIGHT synthesis (graded excitation) ===\n");
      if ((sy = straight_synth_tb06ca(spg, f0info->vector[0], cond.fs, cond.shiftm, cond.sigp, cond.pc, cond.fc, cond.sc,
                                      cond.gdbw, cond.delsp, cond.cornf, cond.delfrac, ap, NODATA, cond.bap_flag, XTRUE, XFALSE, 
                                      XTRUE, cond.df_flag)) == NODATA){
        fprintf(stderr, "straight: straight synth failed\n");
        exit(1);
      } else {
        if (cond.msg_flag == XTRUE)
          fprintf(stderr, "straight synthesis done\n");
      }

      /* memory free */
      xdmfree(ap);

    } else {
      
      /* No aperiodicity files --> Simple exciation */

      /* f0 variation (voiced / unvoiced) */
      f0info->vector[1] = xdvalloc(f0info->vector[0]->length);
      for (k = 0; k < f0info->vector[0]->length; k++) {
        if (f0info->vector[0]->data[k] != 0.0) {
          f0info->vector[1]->data[k] = 0.0;
        } else {
          f0info->vector[1]->data[k] = 1.0;
        }
      }
      /* synthesize speech */
      if (cond.msg_flag == XTRUE)
        fprintf(stderr, "    === STRAIGHT Synthesis ===\n");

      if ((sy = straight_synth_tb06(spg, f0info->vector[0], f0info->vector[1], cond.fs, cond.shiftm, cond.pc, cond.fc, cond.sc,
                                    cond.gdbw, cond.delsp, cond.cornf, cond.delfrac, XTRUE, XFALSE, XTRUE, cond.df_flag)) == NODATA) {
        fprintf(stderr, "straight: straight synth failed\n");
        exit(1);
      } else {
        if (cond.msg_flag == XTRUE)
          fprintf(stderr, "straight synthesis done\n");
      }
    }
    
    /* write wave data */
    xo = xdvtos(sy);
    if (cond.wav_flag == XTRUE) {
      writessignal_wav(options_struct.file[2].name, xo, cond.fs);
    } else {
      writessignal(options_struct.file[2].name, xo, 0);
    }
    if (cond.msg_flag == XTRUE)
      fprintf(stderr, "write wave: %s\n", options_struct.file[2].name);
    
    /* memory free */
    xdmfree(spg);
    xdvfree(sy);
    xdvsfree(f0info);
    xsvfree(xo);
    
    return 0;
}
