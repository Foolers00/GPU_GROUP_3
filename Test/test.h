/////////////////////////////////////////////////////////////////////////////////////
/* TEST_PY */
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
#include "../Data_Types/data_types.h"
#endif

#ifndef GENERAL_FUNCTIONS
#define GENERAL_FUNCTIONS
#include "../Sequential/general_functions.h"
#endif


#ifndef GENERAL_FUNCTIONS_PAR
#define GENERAL_FUNCTIONS_PAR
#include "../Parallel/general_functions_par.h"
#endif

#ifndef PREFIX
#define PREFIX
#include "prefix_scan.h"
#endif

#ifndef MAX_DISTANCE_CUDA
#define MAX_DISTANCE_CUDA
#include "../Parallel/max_distance.h"
#endif

#ifndef MINMAX
#define MINMAX
#include "../Parallel/minmax.h"
#endif



Point* test_sequence_1();

Point_array* test_sequence_2();

Hull* test_sequence_3();

Point_array* test_sequence_4_1();

Hull* test_sequence_4_2(Point_array* points);


// prefix test
void test_sequence_5();


/*
    initializes an array with random numbers with size
*/
void initialData(unsigned long long int *init_array, const size_t size);


/*
    compares prescan results from gpu to sequential one
*/
void compare_prescan_inclusive(unsigned long long int *h_data, unsigned long long int *gpuRef, size_t size);

void compare_prescan_exclusive(unsigned long long int *h_data, unsigned long long int *gpuRef, size_t size);

/*
 * tests the max_distance calculation
 */
void test_max_distance_cuda();

/*
 * tests the minmax calculation
 */
void test_minmax_cuda();

void printData(unsigned long long int *data, const size_t size);