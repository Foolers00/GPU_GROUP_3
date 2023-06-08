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
    Point_array_par* points_below, Line* l);


void split_point_array_side(Point_array_par* points, Point_array_par* points_side, Line* l, int side);


__global__ void setbits(size_t *above_bits, size_t *below_bits, Point* points, Line* l, 
                            size_t array_size, int block_offset);


__global__ void setbits_side(size_t *bits, Point* points, Line* l, 
                            size_t array_size, int side, int block_offset);


__global__ void movevalues(Point* o_data, Point* i_data, size_t* bit_data,
                           size_t* index_data, size_t array_fsize, int block_offset);                        

__device__ void check_point_location_gpu(Line* l, Point z, int* result);

__device__ Vector init_vector_gpu(Point p, Point q, Vector* v);

__device__ void cross_product_gpu(Vector v1, Vector v2, double* result);