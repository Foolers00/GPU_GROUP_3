/////////////////////////////////////////////////////////////////////////////////////
/* MAIN */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef STDIO
#define STDIO
#include <stdio.h>
#endif

#ifndef STDLIB
#define STDLIB
#include <stdlib.h>
#endif

#ifndef STRING
#define STRING
#include <string.h>
#endif

#ifndef GENERAL_FUNCTIONS
#define GENERAL_FUNCTIONS
#include "Sequential/general_functions.h"
#endif

#ifndef GENERAL_FUNCTIONS_PAR
#define GENERAL_FUNCTIONS_PAR
#include "Parallel/general_functions_par.h"
#endif


#ifndef TEST
#define TEST
#include "Test/test.h"
#endif



int main(){

    // Point_array* points = test_sequence_4_1();// generate_random_points(8, -200, 800);
    // Hull* hull = quickhull(points);

    //test_sequence_2();
    //test_sequence_5();

    //test_minmax_cuda();

    validate_minmax();

    return 0;
}