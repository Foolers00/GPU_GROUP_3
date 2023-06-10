/////////////////////////////////////////////////////////////////////////////////////
/* THRUST SPLIT */
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

void thrust_split_point_array(thrust::device_vector<Point>& points, thrust::device_vector<Point>& points_above, 
                              thrust::device_vector<Point>&  points_below, Line l);

void thrust_split_point_array_side(thrust::device_vector<Point>& points, thrust::device_vector<Point>& points_side, 
                                   Line l, int side);

__device__ void check_point_location_thrust(Line* l, Point z, int* result);

__device__ Vector init_vector_thrust(Point p, Point q, Vector* v);

__device__ void cross_product_thrust(Vector v1, Vector v2, double* result);