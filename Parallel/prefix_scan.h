/////////////////////////////////////////////////////////////////////////////////////
/* PREFIX SCAN */
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
    nvidias prescan for one block, added different modes: INCLUSIVE: inclusive scan, EXCLUSIVE: exclusive scan
*/
__device__ void block_prescan(unsigned long long int *g_odata, unsigned long long int *g_idata, unsigned long long int *aux, unsigned long long int *temp, const size_t size);


/*
    splits the blocks and calls nvidias prescan function, the workload states how many chunks one block
    must prefix scan
*/
__global__ void split_prescan(unsigned long long int *g_odata, unsigned long long int *g_idata, unsigned long long int *aux, int* workload);


/*
    prefix scan that can scan upto a size of 1024^3*MAX_BLOCK_COUNT
*/
void master_prescan(unsigned long long int* o_array, unsigned long long int* i_array, size_t array_size, size_t array_bytes, int mode);


/*
    shifts the array by one and inserts at index 0 a zero
*/
__global__ void shift_block(unsigned long long int *o_data, unsigned long long int *t_data, size_t array_size, int* workload);


/*
    adds the values from aux to each corresponding block by one block shift (Block 0 is ignored)
*/
__global__ void add_block(unsigned long long int *o_data, unsigned long long int *aux, size_t array_size, int* workload);


/*
    sequential prefix scan
*/
void seq_prefix_inclusive(unsigned long long int* o_array, unsigned long long int* i_array, size_t array_size);