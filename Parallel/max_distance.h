/////////////////////////////////////////////////////////////////////////////////////
/* MAX DISTANCE CUDA */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef CUDA
#define CUDA
#include <cuda_runtime.h>
#endif

#ifndef TIME
#define TIME
#include <sys/time.h>
#endif

#ifndef GENERAL_FUNCTIONS_PAR
#define GENERAL_FUNCTIONS_PAR
#include "general_functions_par.h"
#endif

#ifndef DATA_TYPES_PAR
#define DATA_TYPES_PAR
#include "../Data_Types/data_types_par.h"
#endif


/*
 * calculates the shortest distance from z to l and stores it at res
 */
__device__ void distance_cuda(Line* l, Point* z, double* res);

/*
 * performs parallel reduction to find the point with max distance to l PER BLOCk. every block stores
 * its result in the max array
 */
__global__ void max_distance_kernel(Line* l, Point* points, int size, Point* max);

/*
 * calculates the point with maximal distance to l. uses several kernel calls.
 */
Point max_distance_cuda(Line l, Point_array* points);
