/////////////////////////////////////////////////////////////////////////////////////
/* SPLIT */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef SPLIT
#define SPLIT
#include "split.h"
#endif

#ifndef GENERAL_FUNCTIONS_PAR
#define GENERAL_FUNCTIONS_PAR
#include "../Parallel/general_functions_par.h"
#endif


void split_point_array(Point_array_par* points, Point_array_par* points_above, 
    Point_array_par* points_below, Line l){

    // workload var
    int grid_size;
    int* array_workload;
    
    int above_grid_size;
    int* above_workload;

    int below_grid_size;
    int* below_workload;

    // sizes
    size_t array_fsize;
    size_t array_fbytes;
    size_t points_above_fsize;
    size_t points_below_fsize;
    size_t points_above_fbytes;
    size_t points_below_fbytes;


    // memory var
    Point* points_gpu;
    unsigned long long int* above_bits;
    unsigned long long int* below_bits;
    unsigned long long int* above_index;
    unsigned long long int* below_index;

    // workload calc
    array_workload = workload_calc(&grid_size, &array_fsize, points->size);
    array_fbytes = array_fsize*sizeof(unsigned long long int);

    // set up memory
    CHECK(cudaMalloc((Point **)&points_gpu, array_fbytes));
    CHECK(cudaMalloc((unsigned long int **)&above_bits, array_fbytes));
    CHECK(cudaMalloc((unsigned long int **)&below_bits, array_fbytes));
    CHECK(cudaMalloc((unsigned long int **)&above_index, array_fbytes));
    CHECK(cudaMalloc((unsigned long int **)&below_index, array_fbytes));

    // transfer point array
    CHECK(cudaMemcpy(points_gpu, points->array, array_fbytes, cudaMemcpyHostToDevice));

    // set bits in above/below array to 1/0 // 0/1 // 0/0
    setbits(above_bits, below_bits, points_gpu, l, array_workload);

    // prefix bits to get indexes
    master_prescan(above_index, above_bits, array_fsize, array_fbytes, EXCLUSIVE);
    master_prescan(below_index, below_bits, array_fsize, array_fbytes, EXCLUSIVE);

    // set size of arrays
    points_above->size = above_index[array_fsize-1]+above_bits[array_fsize-1];
    points_below->size = below_index[array_fsize-1]+below_bits[array_fsize-1];

    // calculate new workload for arrays
    above_workload = workload_calc(&above_grid_size, &points_above_fsize, points_above->size);
    below_workload = workload_calc(&below_grid_size, &points_below_fsize, points_below->size);

    points_above_fbytes = points_above_fsize*sizeof(Point);
    points_below_fbytes = points_below_fsize*sizeof(Point);

    // set up memory for output point arrays
    CHECK(cudaMalloc((unsigned long int **)&points_above->array, points_above_fbytes));
    CHECK(cudaMalloc((unsigned long int **)&points_below->array, points_below_fbytes));

    // move values
    movevalues(points_above->array, points->array, above_bits, above_index, array_workload);
    movevalues(points_below->array, points->array, below_bits, below_index, array_workload);

    // free memory
    CHECK(cudaFree(points_gpu));
    CHECK(cudaFree(above_bits));
    CHECK(cudaFree(below_bits));
    CHECK(cudaFree(above_index));
    CHECK(cudaFree(below_index));

}


void split_point_array_side(Point_array_par* points, Point_array_par* points_side, Line l, int side){

    // workload var
    int grid_size;
    int* array_workload;
    
    int side_grid_size;
    int* side_workload;

    // sizes
    size_t array_fsize;
    size_t array_fbytes;
    size_t points_side_fsize;
    size_t points_side_fbytes;

    // memory var
    Point* points_gpu;
    unsigned long long int* side_bits;
    unsigned long long int* side_index;

    // workload calc
    array_workload = workload_calc(&grid_size, &array_fsize, points->size);
    array_fbytes = array_fsize*sizeof(unsigned long long int);

    // set up memory
    CHECK(cudaMalloc((Point **)&points_gpu, array_fbytes));
    CHECK(cudaMalloc((unsigned long int **)&side_bits, array_fbytes));
    CHECK(cudaMalloc((unsigned long int **)&side_index, array_fbytes));

    // transfer point array
    CHECK(cudaMemcpy(points_gpu, points->array, array_fbytes, cudaMemcpyHostToDevice));

    // set bits in above/below array to 1/0 // 0/1 // 0/0
    setbits_side(side_bits, points_gpu, l, array_workload, side);

    // prefix bits to get indexes
    master_prescan(side_index, side_bits, array_fsize, array_fbytes, EXCLUSIVE);

    // set size of arrays
    points_side->size = side_index[array_fsize-1]+side_bits[array_fsize-1];

    // calculate new workload for arrays
    side_workload = workload_calc(&side_grid_size, &points_side_fsize, points_side->size);

    points_side_fbytes = points_side_fsize*sizeof(Point);

    // set up memory for output point arrays
    CHECK(cudaMalloc((unsigned long int **)&points_side->array, points_side_fbytes));

    // move values
    movevalues(points_side->array, points->array, side_bits, side_index, array_workload);

    // free memory
    CHECK(cudaFree(points_gpu));
    CHECK(cudaFree(side_bits));
    CHECK(cudaFree(side_index));

}


__global__ void setbits(unsigned long long int *above_bits, unsigned long long int *below_bits, Point* points, Line l, 
                            int* workload)
{
    int thid = (blockIdx.x) * blockDim.x + threadIdx.x;

    for(int i = 0; i < workload[blockIdx.x]; i++){

        int index_1 = 2 * thid + i*MAX_BLOCK_COUNT_SHIFT;
        int index_2 = index_1 + 1;

        if(check_point_location(l, points[index_1]) == ABOVE){
            above_bits[index_1] = 1;
            below_bits[index_1] = 0;
        }
        else if(check_point_location(l, points[index_1]) == BELOW){
            above_bits[index_1] = 0;
            below_bits[index_1] = 1;
        }
        else{
            above_bits[index_1] = 0;
            below_bits[index_1] = 0; 
        }

        if(check_point_location(l, points[index_2]) == ABOVE){
            above_bits[index_2] = 1;
            below_bits[index_2] = 0;
        }
        else if(check_point_location(l, points[index_2]) == BELOW){
            above_bits[index_2] = 0;
            below_bits[index_2] = 1;
        }
        else{
            above_bits[index_2] = 0;
            below_bits[index_2] = 0; 
        }
        
    }

}

__global__ void setbits_side(unsigned long long int *bits, Point* points, Line l, 
                            int* workload, int side)
{
    int thid = (blockIdx.x) * blockDim.x + threadIdx.x;

    for(int i = 0; i < workload[blockIdx.x]; i++){

        int index_1 = 2 * thid + i*MAX_BLOCK_COUNT_SHIFT;
        int index_2 = index_1 + 1;

        if(check_point_location(l, points[index_1]) == side){
            bits[index_1] = 1;
        }
        else{
            bits[index_1] = 0;
        }

        if(check_point_location(l, points[index_2]) == side){
            bits[index_2] = 1;
        }
        else{
            bits[index_2] = 0;
        }
        
    }


}



__global__ void movevalues(Point *o_data, Point *i_data, unsigned long long int *bit_data,
                           unsigned long long int *index_data, int* workload)
{

    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int i = 0; i < workload[blockIdx.x]; i++){
        
        int index_1 = 2 * thid + i*MAX_BLOCK_COUNT_SHIFT;
        int index_2 = index_1 + 1;

        if (bit_data[index_1] == 1)
        {
            o_data[index_data[index_1]] = i_data[index_1];
        }

        if (bit_data[index_2] == 1)
        {
            o_data[index_data[index_2]] = i_data[index_2];
        }

    }

}