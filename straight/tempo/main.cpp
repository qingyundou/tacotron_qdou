/*
 *	main.c : main function of TEMPO
 *	V30k18 (matlab)
 *
 *		coded by T. Toda		2001/2/6
 *
 *	TEMPO applied to concatenative TTS
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

#include "tempo.h"

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

typedef struct CONDITION_STRUCT {
    double maxf0;
    double minf0;
    double f0ceil;
    double f0floor;
    double fs;
    double shiftm;
    double f0shiftm;
    char *cf0file;
    char *f0varfile;
    XBOOL cf0_flag;
    XBOOL raw_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {800.0, 40.0, 800.0, 40.0, 16000.0, 1.0, 10.0,
		  NULL, NULL, XFALSE, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[inputfile]", NULL},
    {"[outputfile]", NULL},
};

#define NUM_OPTION 12
OPTION option_struct[] = {
    {"-maxf0", NULL, "maximum of output f0 [Hz]", "maxf0", 
	 NULL, TYPE_DOUBLE, &cond.maxf0, XFALSE},
    {"-minf0", NULL, "minimum of output f0 [Hz]", "minf0", 
	 NULL, TYPE_DOUBLE, &cond.minf0, XFALSE},
    {"-uf0", NULL, "f0 upper limit [Hz]", "upperf0", 
	 NULL, TYPE_DOUBLE, &cond.f0ceil, XFALSE},
    {"-lf0", NULL, "f0 lower limit [Hz]", "lowerf0", 
	 NULL, TYPE_DOUBLE, &cond.f0floor, XFALSE},
    {"-f", NULL, "sampling frequency [Hz]", "samp_freq", 
	 NULL, TYPE_DOUBLE, &cond.fs, XFALSE},
    {"-shift", NULL, "frame shift [ms], (< 5 ms)", "shift", 
	 NULL, TYPE_DOUBLE, &cond.shiftm, XFALSE},
    {"-f0shift", NULL, "f0 frame shift [ms]", "f0shift", 
	 NULL, TYPE_DOUBLE, &cond.f0shiftm, XFALSE},
    {"-cf0file", NULL, "candidate f0 file", "cf0file", 
	 NULL, TYPE_STRING, &cond.cf0file, XFALSE},
//    {"-f0varfile", NULL, "f0 var file", "f0varfile", 
//	 NULL, TYPE_STRING, &cond.f0varfile, XFALSE},
    {"-onlycf0", NULL, "only output candidate f0 file", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.cf0_flag, XFALSE},
    {"-raw", NULL, "input raw file (16bit short)", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.raw_flag, XFALSE},
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
    int i;
    int fc;
    SVECTOR xs = NODATA;
    DVECTOR xd = NODATA;
    DVECTORS f0info = NODATA;
    
    /* get program name */
    options_struct.progname = xgetbasicname(argv[0]);

    /* get option */
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);

    /* display message */
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "Time-domain Excitation extractor using Minimum Perturbation Operator");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");
    
    /* read wave data */
    if (cond.raw_flag == XFALSE) {
	xd = xread_wavfile(options_struct.file[0].name, &cond.fs,
			  cond.msg_flag);
    } else {
	if ((xs = xreadssignal(options_struct.file[0].name, 0, 0)) == NODATA) {
	    fprintf(stderr, "Can't read wave data\n");
	    exit(1);
	} else {
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "read signal: %s\n",
			options_struct.file[0].name);
	    xd = xsvtod(xs);
	    xsvfree(xs);
	}
    }
    if (0) {
	xs = xdvtos(xd);
	writessignal_wav(options_struct.file[1].name, xs, cond.fs);
	xdvfree(xd);
	xsvfree(xs);
	exit(1);
    }

    if (!strnone(cond.f0varfile)) {
	if ((f0info = tempo(xd, cond.maxf0, cond.minf0, cond.f0ceil,
			    cond.f0floor, cond.fs, cond.shiftm, cond.f0shiftm,
			    cond.cf0file, XTRUE, XFALSE, cond.msg_flag,
			    cond.cf0_flag)) == NODATA) {
	    fprintf(stderr, "Failed pitch extraction\n");
	    exit(1);
	}
	/* write f0var file */
	writedvector_txt(cond.f0varfile, f0info->vector[1]);
    } else {
	if ((f0info = tempo(xd, cond.maxf0, cond.minf0, cond.f0ceil,
			    cond.f0floor, cond.fs, cond.shiftm, cond.f0shiftm,
			    cond.cf0file, XFALSE, XFALSE, cond.msg_flag,
			    cond.cf0_flag)) == NODATA) {
	    fprintf(stderr, "Failed pitch extraction\n");
	    exit(1);
	}
    }
    
    /* write f0 file */
    writedvector_txt(options_struct.file[1].name, f0info->vector[0]);

    /* memory free */
    xdvfree(xd);
    xdvsfree(f0info);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
