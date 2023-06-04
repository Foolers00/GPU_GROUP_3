/////////////////////////////////////////////////////////////////////////////////////
/* SPLIT */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef PREFIX
#define PREFIX
#include "prefix_scan.h"
#endif

#ifndef GENERAL_FUNCTIONS_PAR
#define GENERAL_FUNCTIONS_PAR
#include "general_functions_par.h"
#endif



void split_point_array(Point_array_par* points, Point_array_par* points_above, 
    Point_array_par* points_below, Line l);


void split_point_array_side(Point_array_par* points, Point_array_par* points_side, Line l, int side);


__global__ void setbits(unsigned long long int *above_bits, unsigned long long int *below_bits, 
                        Point* points, Line l, int* workload);


__global__ void setbits_side(unsigned long long int *bits, Point* points, Line l, 
                            int* workload, int side);


__global__ void movevalues(unsigned int *o_data, unsigned int *i_data, unsigned int *bit_data,
                           unsigned int *index_data, const int offset, int* workload);                            