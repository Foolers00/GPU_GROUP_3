/////////////////////////////////////////////////////////////////////////////////////
/* THRUST MAX DISTANCE */
/////////////////////////////////////////////////////////////////////////////////////

#define THRUST_SPLIT

#ifndef DATA_TYPES_PAR
#define DATA_TYPES_PAR
#include "../Data_Types/data_types_par.h"
#endif


#ifndef VECTOR
#define VECTOR
#include "../Data_Types/vector.h"
#endif


#ifndef THRUST
#define THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#endif

__device__ void dist(Line* l, const Point& z, double& res);
void thrust_max_distance(thrust::device_vector<Line>& l, thrust::device_vector<Point>& points, thrust::device_vector<Line>& l_max);