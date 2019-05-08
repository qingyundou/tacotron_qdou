/*
 *	main.c : main function of straight
 *
 *		coded by H.Banno 	1996/12/25
 *		modified by T. Toda	2001/2/12
 *			V30k18 (matlab)
 *
 *	STRAIGHT Analysis applied to concatenative TTS
 *		coded by T. Toda
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
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
    double fs;		/* Sampling frequency [Hz] */
    double shiftm;	/* Frame shift (ms) */
    double f0shiftm;	/* Frame shift (ms) */
    long fftl;		/* FFT length */
    long order;		/* Cepstrum order */
    long apord;		/* IFFT Aperiodic order */
    char *f0file;	/* F0 text file */
    char *cf0file;	/* candidate F0 text file */
    char *apfile;	/* aperiodic energy file */
    char *npowfile;	/* normalized power file */
    XBOOL cep_flag;	/* Cepstrogram */
    XBOOL mcep_flag;	/* Mel Cepstrogram */
    double alpha;	/* Frequency warping factor for all-pass filter used for mel-cepstrum */
    XBOOL pow_flag;	/* include Power coefficient */
    XBOOL lpow_flag;	/* include Power coefficient */
    XBOOL wpow_flag;	/* include Power coefficient */
    XBOOL fast_flag;	/* fast analysis version */
    XBOOL wav_flag;	/* wav file */
    XBOOL float_flag;	/* float */
    XBOOL msg_flag;	/* print message */
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {16000.0, 5.0, 10.0, 1024, 24, 513, NULL, NULL, NULL, NULL,
                  XFALSE, XFALSE, 0.42, XFALSE, XFALSE, XFALSE, XFALSE, XTRUE,
                  XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[inputfile]", NULL},
    {"[outputfile]", NULL},
};

#define NUM_OPTION 20
OPTION option_struct[] = {
    {"-f", NULL, "sampling frequency [Hz]", "samp_freq", 
	 NULL, TYPE_DOUBLE, &cond.fs, XFALSE},
    {"-shift", NULL, "frame shift [ms]", "shift", 
	 NULL, TYPE_DOUBLE, &cond.shiftm, XFALSE},
    {"-f0shift", NULL, "F0 frame shift [ms]", "f0shift", 
	 NULL, TYPE_DOUBLE, &cond.f0shiftm, XFALSE},
    {"-fftl", NULL, "fft length", "fft_length", 
	 NULL, TYPE_LONG, &cond.fftl, XFALSE},
    {"-order", NULL, "cepstrum order", "order", 
	 NULL, TYPE_LONG, &cond.order, XFALSE},
    {"-apord", NULL, "IFFT aperiodic energy order", "apord", 
	 NULL, TYPE_LONG, &cond.apord, XFALSE},
    {"-f0file", NULL, "F0 filename", "f0file", 
	 NULL, TYPE_STRING, &cond.f0file, XFALSE},
    {"-cf0file", NULL, "candidate F0 filename", "cf0file", 
	 NULL, TYPE_STRING, &cond.cf0file, XFALSE},
    {"-apfile", NULL, "aperiodic energy filename", "apfile", 
	 NULL, TYPE_STRING, &cond.apfile, XFALSE},
    {"-npowfile", NULL, "normalized power filename", "npowfile", 
	 NULL, TYPE_STRING, &cond.npowfile, XFALSE},
    {"-cep", NULL, "cepstrogram", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.cep_flag, XFALSE},
    {"-mcep", NULL, "mel cepstrogram", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.mcep_flag, XFALSE},
    {"-alpha", NULL, "warping factor", "alpha", 
	 NULL, TYPE_DOUBLE, &cond.alpha, XFALSE},
    {"-pow", NULL, "include power coefficient", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.pow_flag, XFALSE},
    {"-lpow", NULL, "include linear power coefficient", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.lpow_flag, XFALSE},
    {"-wpow", NULL, "include waveform power coefficient", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.wpow_flag, XFALSE},
    {"-fast", NULL, "using fast analysis version", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.fast_flag, XFALSE},
    {"-raw", NULL, "input raw file (16bit short)", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.wav_flag, XFALSE},
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
    long ord;
    double eta, pc, mag;
    SVECTOR x = NODATA;
    DVECTOR xd = NODATA;
    DVECTOR f0l = NODATA;
    DVECTOR if0l = NODATA;
    DVECTOR powvec = NODATA;
    DMATRIX n2sgram = NODATA;
    DMATRIX dapv = NODATA;
    DMATRIX idapv = NODATA;
    DMATRIX cepg = NODATA;
    DMATRIX mcepg = NODATA;

    /*  
    struct tms t0;
    struct tms t1;
    time_t tck0;
    time_t tck1;
    // start time
    tck0 = times(&t0);
    */

    eta = 1.4;
    pc = 0.6;
    mag = 0.2;	/* This parameter should be revised */
    
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
	xd = xread_wavfile(options_struct.file[0].name, &cond.fs,
			   cond.msg_flag);
    } else {
	/* read wave data */
	if ((x = xreadssignal(options_struct.file[0].name, 0, 0)) == NODATA) {
	    fprintf(stderr, "Can't read wave data\n");
	    exit(1);
	} else {
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "read wave: %s\n",
			options_struct.file[0].name);
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
	fprintf(stderr, "F0shift %f[ms] => %f[ms]\n",
		cond.f0shiftm, cond.shiftm);
    if0l = xget_interp_f0(f0l, cond.f0shiftm, cond.shiftm);
    // memory free
    xdvfree(f0l);

    /* ANALYSIS */
    if (cond.msg_flag == XTRUE) fprintf(stderr, "     === STRAIGHT-Analysis ===\n");
    /* calculate spectrogram */
    if ((n2sgram = straight_body_c03(xd, if0l, cond.fs, 40.0, cond.shiftm,
				     cond.fftl, eta, pc, cond.fast_flag,
				     cond.msg_flag)) == NODATA) {
	fprintf(stderr, "straight: straight body failed\n");
	exit(1);
    } else {
	if (cond.fast_flag == XFALSE)
	    specreshape(n2sgram, if0l, cond.fs, eta, pc, mag, cond.msg_flag);
	/* if (cond.msg_flag == XTRUE) fprintf(stderr, "straight body\n"); */
    }
    cond.fftl = (n2sgram->col - 1) * 2;

    if (!strnone(cond.apfile)) {
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "--- MBE type analysis ---\n");
	/* calculate aperiodic energy */
	if ((dapv = aperiodicpart4(xd, if0l, cond.fs, cond.shiftm, 1.0,
				   cond.fftl / 2 + 1, cond.msg_flag)) == NODATA) {
	    fprintf(stderr, "straight: MBE type analysis failed\n");
	    exit(1);
	} else {
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "         relative aperiodic energy\n");
	}
    }

    /* memory free */
    xdvfree(if0l);

    /* temporary */
    /* normalized power sequence */
    if (!strnone(cond.npowfile)) {
        powvec = xspg2pow_norm(n2sgram);
        if (cond.float_flag == XFALSE) writedsignal(cond.npowfile, powvec, 0);
        else writed2fsignal(cond.npowfile, powvec, 0);
        if (cond.msg_flag == XTRUE)
            fprintf(stderr, "write normalized power file [%ld]: %s\n",
                    powvec->length, cond.npowfile);
        /* memory free */
        xdvfree(powvec);
    }

    /* write analysis file */
    if (cond.cep_flag == XFALSE && cond.mcep_flag == XFALSE) {
	if (cond.float_flag == XFALSE) {
	    writedmatrix(options_struct.file[1].name, n2sgram, 0);
	} else {
	    writed2fmatrix(options_struct.file[1].name, n2sgram, 0);
	}
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "write spectrogram [%ld][%ld]: %s\n",
		    n2sgram->row, n2sgram->col, options_struct.file[1].name);
	/* memory free */
	xdmfree(n2sgram);
    } else {
	/* change spectrogram into cepstrogram */
	ord = cond.order;
	if (cond.mcep_flag == XTRUE) ord = n2sgram->col - 1;
	if (cond.lpow_flag == XTRUE || cond.wpow_flag == XTRUE)
	    cond.pow_flag = XTRUE;
	cepg = xget_spg2cepg(n2sgram, ord, cond.pow_flag);
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "change Spectrogram [%ld][%ld] into Cepstrogram [%ld][%ld]\n", n2sgram->row, n2sgram->col, cepg->row, cepg->col);
	if (cond.lpow_flag == XTRUE) {
	    powvec = xget_spg2powvec(n2sgram, XTRUE);
	} else if (cond.wpow_flag == XTRUE) {
	    powvec = xget_wave2powvec(xd, cond.fs, 20.0, cond.shiftm,
				      n2sgram->row, XTRUE);
	}
	/* memory free */
	xdmfree(n2sgram);

	if (cond.mcep_flag == XFALSE) {
	    if (cond.lpow_flag == XTRUE || cond.wpow_flag == XTRUE) {
		if (cond.msg_flag == XTRUE)
		    fprintf(stderr, "substitute to power on linear domain\n");
		dmpastecol(cepg, 0, powvec, 0, powvec->length, 0);
		/* memory free */
		xdvfree(powvec);
	    }
	    if (cond.float_flag == XFALSE) {
		writedmatrix(options_struct.file[1].name, cepg, 0);
	    } else {
		writed2fmatrix(options_struct.file[1].name, cepg, 0);
	    }
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "write cepstrogram [%ld][%ld]: %s\n",
			cepg->row, cepg->col, options_struct.file[1].name);
	} else {
	    /* change linear scale into Mel scale */
      mcepg = xcepg2mpmcepg(cepg, cond.order, cond.fftl, cond.pow_flag,
                            XFALSE, cond.alpha);
	    if (cond.lpow_flag == XTRUE || cond.wpow_flag == XTRUE) {
		if (cond.msg_flag == XTRUE)
		    fprintf(stderr, "substitute to power on linear domain\n");
		dmpastecol(mcepg, 0, powvec, 0, powvec->length, 0);
		/* memory free */
		xdvfree(powvec);
	    }
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "change Cepstrogram [%ld][%ld] into Mel Cepstrogram [%ld][%ld]\n", cepg->row, cepg->col, mcepg->row, mcepg->col);
	    if (cond.float_flag == XFALSE) {
		writedmatrix(options_struct.file[1].name, mcepg, 0);
	    } else {
		writed2fmatrix(options_struct.file[1].name, mcepg, 0);
	    }
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "write Mel cepstrogram [%ld][%ld]: %s\n",
			mcepg->row, mcepg->col, options_struct.file[1].name);

	    /*	    
	    // end time
	    tck1 = times(&t1);
	    printf("# number of frames = %ld\n", mcepg->row);
	    printf("# elapsed_time = %f\n", (float)(tck1 - tck0)
		   / sysconf(_SC_CLK_TCK));
	    printf("# use_time     = %f\n",
		   (float)(t1.tms_utime - t0.tms_utime) / CLK_TCK);
	    printf("# system_time  = %f\n",
		   (float)(t1.tms_stime - t0.tms_stime) / CLK_TCK);
	    */

	    /* memory free */
	    xdmfree(mcepg);
	}
	/* memory free */
	xdmfree(cepg);
    }
    if (!strnone(cond.apfile)) {
	if (cond.apord < cond.fftl / 2 + 1) {
	    idapv = xget_fftmat(dapv, cond.fftl, cond.apord, XTRUE);
	    /* write aperiodic energy file */
	    if (cond.float_flag == XFALSE) {
		writedmatrix(cond.apfile, idapv, 0);
	    } else {
		writed2fmatrix(cond.apfile, idapv, 0);
	    }
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "write IFFT aperiodic energy [%ld][%ld]: %s\n",
			idapv->row, idapv->col, cond.apfile);
	    /* memory free */
	    xdmfree(idapv);
	} else {
	    /* write aperiodic energy file */
	    if (cond.float_flag == XFALSE) {
		writedmatrix(cond.apfile, dapv, 0);
	    } else {
		writed2fmatrix(cond.apfile, dapv, 0);
	    }
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "write aperiodic energy [%ld][%ld]: %s\n",
			dapv->row, dapv->col, cond.apfile);
	}
	/* memory free */
	xdmfree(dapv);
    }
    /* memory free */
    xdvfree(xd);

    return 0;
}
