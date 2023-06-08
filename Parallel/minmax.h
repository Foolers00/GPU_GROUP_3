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

typedef struct{
    //one array for finding min, one for finding max
    Point *min, *max;
}minmaxPoint;

/*
 * writes the point with minimal x value in points to min and with maximal x value to max.
 * Same as points_on_hull in sequential. Uses several kernel calls.
 */
void minmax_cuda(Point_array_par* points, Point* min, Point* max);

/*
 * returns line l_pq gpu mem adress where l_pq.p is min and l_pq.q is max
 * Same as points_on_hull in sequential. Uses several kernel calls.
*/
//void minmax_cuda(Point_array_par* points, Line** l_pq);

/*
 *
 */
__global__ void minmax_kernel(minmaxPoint points, int size, minmaxPoint result);