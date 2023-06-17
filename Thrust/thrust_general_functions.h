/////////////////////////////////////////////////////////////////////////////////////
/* THRUST GENERAL FUNCTIONS */
/////////////////////////////////////////////////////////////////////////////////////



#define THRUST_GENERAL_FUNCTIONS

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


#ifndef THRUST_SPLIT
#define THRUST_SPLIT
#include "thrust_split.h"
#endif

#ifndef THRUST_MAX_DISTANCE
#define THRUST_MAX_DISTANCE
#include "thrust_max_distance.h"
#endif


#ifndef THRUST_MINMAX
#define THRUST_MINMAX
#include "thrust_minmax.h"
#endif




void thrust_quickhull(Point_array_par* points, thrust::host_vector<Line>& hull);


void thrust_first_quickhull_split(thrust::device_vector<Point>& points, thrust::device_vector<Line>& hull,
                                  Line l, int side);


void thrust_combine_hull(thrust::device_vector<Line>& hull_up, thrust::device_vector<Line>& hull_down, 
                         thrust::device_vector<Line>& hull_result);



