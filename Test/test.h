/////////////////////////////////////////////////////////////////////////////////////
/* TEST_PY */
/////////////////////////////////////////////////////////////////////////////////////

#define TEST

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

#ifndef SPLIT
#define SPLIT
#include "../Parallel/split.h"
#endif



Point* test_sequence_1();


Point_array* test_sequence_2();


Hull* test_sequence_3();


Point_array* test_sequence_4_1();


Hull* test_sequence_4_2(Point_array* points);


// prefix test
void test_prescan();
void test_prescan_gpu();


/*
    initializes an array with random numbers with size
*/
void initialData(size_t *init_array, const size_t size);


/*
    compares prescan results from gpu to sequential one
*/
void compare_prescan_inclusive(size_t *h_data, size_t *gpuRef, size_t size);


void compare_prescan_exclusive(size_t *h_data, size_t *gpuRef, size_t size);


/*
 * tests the max_distance calculation
 */
void test_max_distance_cuda();


/*
 * tests the minmax calculation
 */
void test_minmax_cuda();


/*
 * compares the results of minmax to the results of sequential implementation
 */
void validate_minmax();

/*
    test the check above split
*/
void test_split();


void printData(size_t *data, const size_t size);