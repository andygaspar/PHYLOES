//
//  Copyright 2002-2007 Rick Desper, Olivier Gascuel
//  Copyright 2007-2014 Olivier Gascuel, Stephane Guindon, Vincent Lefort
//
//  This file is part of FastME.
//
//  FastME is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  FastME is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with FastME.  If not, see <http://www.gnu.org/licenses/>
//

#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <getopt.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <libgen.h>
#include <assert.h>
#include <omp.h>



void constantToStr (int c, char *str);
int *initZeroArray (int l);
int *initOneArray (int l);
double **initDoubleMatrix (int d);
void fillZeroMatrix (double ***A, int d);
void deleteMatrix (double **D, int size);

