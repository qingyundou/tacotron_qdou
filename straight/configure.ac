#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT(STRAIGHT, 1.0, BUG-REPORT-ADDRESS)
AC_CONFIG_SRCDIR([sp/complex.cpp])
PACKAGE=STRAIGHT
VERSION=1.0
AM_INIT_AUTOMAKE($PACKAGE,$VERSION,no-define)


# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_RANLIB
AC_PROG_INSTALL

# Checks for libraries.

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([stdio.h stddef.h stdlib.h string.h sys/stat.h sys/types.h math.h varargs.h stdarg.h])


# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_TYPE_SIZE_T
AC_STRUCT_TM


# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STAT
AC_FUNC_VPRINTF
AC_CHECK_FUNCS([floor modf pow sqrt strchr strrchr])

AC_CONFIG_FILES([Makefile sp/Makefile tempo/Makefile analysis/Makefile synthesis/Makefile])

AC_OUTPUT
