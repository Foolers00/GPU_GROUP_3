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

#ifndef DATA_TYPES
#define DATA_TYPES
#include "data_types.h"
#endif

#ifndef GENERAL_FUNCTIONS
#define GENERAL_FUNCTIONS
#include "general_functions.h"
#endif


#ifndef DYNAMIC_ARRAY
#define DYNAMIC_ARRAY
#include "dynamic_array.h"
#endif

#ifndef TEST
#define TEST
#include "test.h"
#endif



int main(){
    //Point_array* points = test_sequence_1();
    Point_array* points = test_sequence_1();
    print_point_array(points);
    return 0;
}