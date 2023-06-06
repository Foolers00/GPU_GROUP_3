/////////////////////////////////////////////////////////////////////////////////////
/* SPLIT */
/////////////////////////////////////////////////////////////////////////////////////

#define SPLIT

#ifndef GENERAL_FUNCTIONS
#define GENERAL_FUNCTIONS
#include "../Sequential/general_functions.h"
#endif

#ifndef PREFIX
#define PREFIX
#include "prefix_scan.h"
#endif

#ifndef DATA_TYPES_PAR
#define DATA_TYPES_PAR
#include "../Data_Types/data_types_par.h"
#endif



void split_point_array(Point_array_par* points, Point_array_par* points_above, 
    Point_array_par* points_below, Line l);


void split_point_array_side(Point_array_par* points, Point_array_par* points_side, Line l, int side);


__global__ void setbits(unsigned long long int *above_bits, unsigned long long int *below_bits, 
                        Point* points, Line l, int* workload);


__global__ void setbits_side(unsigned long long int *bits, Point* points, Line l, 
                            int* workload, int side);


__global__ void movevalues(Point *o_data, Point *i_data, unsigned long long int *bit_data,
                           unsigned long long int *index_data, int* workload);                        

__device__ void check_point_location_gpu(Line l, Point z, int* result);

__device__ Vector init_vector_gpu(Point p, Point q, Vector* v);

__device__ void cross_product_gpu(Vector v1, Vector v2, double* result);