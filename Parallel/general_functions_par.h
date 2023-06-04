/////////////////////////////////////////////////////////////////////////////////////
/* GENERAL FUNCTIONS PARALLEL */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef CUDA
#define CUDA
#include <cuda_runtime.h>
#endif

#ifndef TIME
#define TIME
#include <sys/time.h>
#endif

#ifndef GENERAL_FUNCTIONS
#define GENERAL_FUNCTIONS
#include "../Sequential/general_functions.h"
#endif


#ifndef DATA_TYPES_PAR
#define DATA_TYPES_PAR
#include "../Data_Types/data_types_par.h"
#endif


#ifndef PREFIX
#define PREFIX
#include "prefix_scan.h"
#endif





Hull* quickhull_split_par(Point_array* points, Line l, int side);





int* workload_calc(int *grid_size, size_t* array_fsize, size_t array_size);