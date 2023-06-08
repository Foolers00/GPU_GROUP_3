/////////////////////////////////////////////////////////////////////////////////////
/* GENERAL FUNCTIONS PARALLEL */
/////////////////////////////////////////////////////////////////////////////////////

#define GENERAL_FUNCTIONS_PAR

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

#ifndef MINMAX
#define MINMAX
#include "minmax.h"
#endif

#ifndef MAX_DISTANCE
#define MAX_DISTANCE
#include "max_distance.h"
#endif


Hull_par* quickhull_par(Point_array_par* points);


Hull_par* quickhull_split_par(Point_array_par* points, Line* l, int side);

void workload_calc(size_t* grid_size, size_t* rem_grid_size, size_t* loop_cnt, size_t* sizef, size_t size);

Point_array_par* generate_random_points_par(int num_of_points, double l_bound, double u_bound);


Hull_par* combine_hull_par(Hull_par* hull_1, Hull_par* hull_2);

void points_on_hull_par(Point_array_par* points, Line** l_pq);
