/////////////////////////////////////////////////////////////////////////////////////
/* MAX DISTANCE CUDA */
/////////////////////////////////////////////////////////////////////////////////////

#define MINMAX

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
 * writes the point with minimal x value in points to min and with maximal x value to max.
 * Same as points_on_hull in sequential. Uses several kernel calls.
 */
void minmax_cuda(Point_array* points, Point* min, Point* max);

/*
 *
 */
__global__ void minmax_kernel(Point* points, int size, Point* max, int op);