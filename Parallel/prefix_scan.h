/////////////////////////////////////////////////////////////////////////////////////
/* PREFIX SCAN */
/////////////////////////////////////////////////////////////////////////////////////

#define PREFIX

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




/*
    nvidias prescan for one block, added different modes: INCLUSIVE: inclusive scan, EXCLUSIVE: exclusive scan
*/
__device__ void block_prescan(size_t *g_odata, size_t *g_idata, size_t *aux, const size_t size);


/*
    splits the blocks and calls nvidias prescan function, the workload states how many chunks one block
    must prefix scan
*/
__global__ void split_prescan(size_t* g_odata, size_t* g_idata, size_t* aux, int block_offset);


/*
    prefix scan that can scan upto a size of 1024^3*MAX_BLOCK_COUNT, i_array and o_array are cpu memories with size
    array_size and bytes array_bytes, mode: INCLUSIVE/EXCLUSIVE
*/
void master_prescan(size_t* o_array, size_t* i_array, size_t array_size, size_t array_bytes, int mode);
void master_stream_prescan(size_t* o_array, size_t* i_array, size_t array_size, size_t array_bytes, int mode);

/*
    prefix scan that can scan upto a size of 1024^3*MAX_BLOCK_COUNT, i_array and o_array are gpu memories with size
    array_fsize and bytes array_fbytes, array_workload states the workload of every block, mode: INCLUSIVE/EXCLUSIVE
*/
void master_prescan_gpu(size_t* o_array, size_t* i_array, size_t array_fsize, size_t array_fbytes, 
                            int array_grid_size, int array_rem_grid_size, int array_loop_cnt, int mode);

void master_stream_prescan_gpu(size_t* o_array, size_t* i_array, size_t array_fsize, size_t array_fbytes, 
                            int array_grid_size, int array_rem_grid_size, int array_loop_cnt, int mode);

/*
    shifts the array by one and inserts at index 0 a zero
*/
__global__ void shift_block(size_t* o_data, size_t* t_data, size_t array_size, int block_offset);


/*
    adds the values from aux to each corresponding block by one block shift (Block 0 is ignored)
*/
__global__ void add_block(size_t* o_data, size_t* aux, int block_offset);


/*
    sequential prefix scan
*/
void seq_prefix_inclusive(size_t* o_array, size_t* i_array, size_t array_size);










