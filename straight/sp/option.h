/* option.h
 *	coded by H. Banno
 *
 *	Tomoki Toda (tomoki.toda@atr.co.jp)
 *			From Mar. 2001 to Sep. 2003
 */

#ifndef __OPTION_H
#define __OPTION_H

#include "defs.h"

#define TYPE_BOOLEAN 0
#define TYPE_STRING 1
#define TYPE_INT 2
#define TYPE_SHORT 3
#define TYPE_LONG 4
#define TYPE_FLOAT 5
#define TYPE_DOUBLE 6
#define TYPE_CHAR 7
#define TYPE_OTHERS 8
#define TYPE_NONE 9
#define TYPE_XRM 10
#define TYPE_STRING_S 11

#define USAGE_LABEL_STRING "???"

#define eqtype(type1, type2) ((type1 == type2) ? 1 : 0)
#define str2bool(value) (((*(value) == 'T') || streq(value, "ON") || streq(value, "On")) ? XTRUE : XFALSE)
#define bool2str(value) ((*(XBOOL *)(value) == XTRUE) ? "True" : "False")

typedef struct ARGFILE_STRUCT {
    char *label;	/* label for help message */
    char *name;		/* filename */
} ARGFILE, *ARGFILE_P;

typedef struct OPTION_STRUCT {
    char *flag;		/* option flag */
    char *subflag;	/* option subflag */
    char *desc;		/* description for help */
    char *label;	/* label for setup file */
    char *specifier;	/* specifier for using X11 */
    int type;		/* type of value */
    void *value;	/* option value */
    XBOOL changed;	/* true if value changed */
} OPTION, *OPTION_P;

typedef struct OPTIONS_STRUCT {
    char *progname;	/* program name */
    int section;	/* section number */
    int num_option;	/* number of option */
    OPTION *option;	/* option structure */
    int num_file;	/* number of file */
    ARGFILE *file;	/* file structure */
} OPTIONS, *OPTIONS_P;

extern char *getbasicname(char *name);
extern char *xgetbasicname(char *name);
extern char *xgetdirname(char *name);
extern char *xgetexactname(char *name);
extern int flageq(char *flag, OPTIONS options);
extern int convoptvalue(char *value, OPTION *option);
extern int setoptvalue(char *value, OPTION *option);
extern int getoption(int argc, char *argv[], int *ac, OPTIONS *options);
extern void setchanged(int argc, char *argv[], OPTIONS *options);
extern int getargfile(char *filename, int *fc, OPTIONS *options);
#ifdef VARARGS
extern void printhelp();
extern void printerr();
#else
extern void printhelp(OPTIONS options, char *format, ...);
extern void printerr(OPTIONS options, char *format, ...);
#endif
extern int labeleq(char *label, OPTIONS *options);
extern void readsetup(char *filename, OPTIONS *options);
extern void writesetup(char *filename, OPTIONS options);
extern void usage(OPTIONS options);

#endif /* __OPTION_H */
