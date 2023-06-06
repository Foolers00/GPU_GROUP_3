/////////////////////////////////////////////////////////////////////////////////////
/* SPLIT */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef SPLIT
#define SPLIT
#include "split.h"
#endif


void split_point_array(Point_array_par* points, Point_array_par* points_above, 
    Point_array_par* points_below, Line l){

    // workload var
    int array_grid_size;
    int* array_workload;

    // sizes
    size_t array_fsize;
    size_t array_fbytes;
    size_t points_above_bytes;
    size_t points_below_bytes;


    // memory var
    Point* points_gpu;
    unsigned long long int* above_bits;
    unsigned long long int* below_bits;
    unsigned long long int* above_index;
    unsigned long long int* below_index;

    // workload calc
    array_workload = workload_calc(&array_grid_size, &array_fsize, points->size);
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
    setbits<<<array_grid_size, BLOCKSIZE>>>(above_bits, below_bits, points_gpu, l, array_workload);

    // prefix bits to get indexes
    master_prescan(above_index, above_bits, array_fsize, array_fbytes, EXCLUSIVE);
    master_prescan(below_index, below_bits, array_fsize, array_fbytes, EXCLUSIVE);

    // set size of arrays
    points_above->size = above_index[array_fsize-1]+above_bits[array_fsize-1];
    points_below->size = below_index[array_fsize-1]+below_bits[array_fsize-1];

    // calculate new workload for arrays
    points_above_bytes = points_above->size*sizeof(Point);
    points_below_bytes = points_below->size*sizeof(Point);

    // set up memory for output point arrays
    CHECK(cudaMalloc((unsigned long int **)&points_above->array, points_above_bytes));
    CHECK(cudaMalloc((unsigned long int **)&points_below->array, points_below_bytes));

    // move values
    movevalues<<<array_grid_size, BLOCKSIZE>>>(points_above->array, points->array, above_bits, above_index, array_workload);
    movevalues<<<array_grid_size, BLOCKSIZE>>>(points_below->array, points->array, below_bits, below_index, array_workload);

    // free memory
    CHECK(cudaFree(points_gpu));
    CHECK(cudaFree(above_bits));
    CHECK(cudaFree(below_bits));
    CHECK(cudaFree(above_index));
    CHECK(cudaFree(below_index));

}


void split_point_array_side(Point_array_par* points, Point_array_par* points_side, Line l, int side){

    // workload var
    int array_grid_size;
    int* array_workload;

    // sizes
    size_t array_fsize;
    size_t array_fbytes;
    size_t points_side_bytes;

    // memory var
    Point* points_gpu;
    unsigned long long int* side_bits;
    unsigned long long int* side_index;

    // workload calc
    array_workload = workload_calc(&array_grid_size, &array_fsize, points->size);
    array_fbytes = array_fsize*sizeof(unsigned long long int);

    // set up memory
    CHECK(cudaMalloc((Point **)&points_gpu, array_fbytes));
    CHECK(cudaMalloc((unsigned long int **)&side_bits, array_fbytes));
    CHECK(cudaMalloc((unsigned long int **)&side_index, array_fbytes));

    // transfer point array
    CHECK(cudaMemcpy(points_gpu, points->array, array_fbytes, cudaMemcpyHostToDevice));

    // set bits in above/below array to 1/0 // 0/1 // 0/0
    setbits_side<<<array_grid_size, BLOCKSIZE>>>(side_bits, points_gpu, l, array_workload, side);

    // prefix bits to get indexes
    master_prescan(side_index, side_bits, array_fsize, array_fbytes, EXCLUSIVE);

    // set size of arrays
    points_side->size = side_index[array_fsize-1]+side_bits[array_fsize-1];
    points_side_bytes = points_side->size*sizeof(Point);

    // set up memory for output point arrays
    CHECK(cudaMalloc((unsigned long int **)&points_side->array, points_side_bytes));

    // move values
    movevalues<<<array_grid_size, BLOCKSIZE>>>(points_side->array, points->array, side_bits, side_index, array_workload);

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
        int result;

        check_point_location_gpu(l, points[index_1], &result);

        if(result == ABOVE){
            above_bits[index_1] = 1;
            below_bits[index_1] = 0;
        }
        else if(result == BELOW){
            above_bits[index_1] = 0;
            below_bits[index_1] = 1;
        }
        else{
            above_bits[index_1] = 0;
            below_bits[index_1] = 0; 
        }

        check_point_location_gpu(l, points[index_2], &result);

        if(result == ABOVE){
            above_bits[index_2] = 1;
            below_bits[index_2] = 0;
        }
        else if(result == BELOW){
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
        int result;

        check_point_location_gpu(l, points[index_1], &result);

        if(result == side){
            bits[index_1] = 1;
        }
        else{
            bits[index_1] = 0;
        }

        check_point_location_gpu(l, points[index_2], &result);

        if(result == side){
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


__device__ void check_point_location_gpu(Line l, Point z, int* result){

    Vector v1;
    Vector v2; 
    
    double cross_result;

    init_vector_gpu(l.p, l.q, &v1);
    init_vector_gpu(l.p, z, &v2);
    
    cross_product_gpu(v1, v2, &cross_result);

    if(cross_result>0){
        *result = ABOVE;
        return;
    }
    if(cross_result == 0){
        *result = ON;
        return;
    }
    *result = BELOW;
    return;

}

__device__ Vector init_vector_gpu(Point p, Point q, Vector* v){

    v->_x = q.x-p.x;
    v->_y = q.y-p.y;

    return;
}


__device__ void cross_product_gpu(Vector v1, Vector v2, double* result){
    
    *result = (v1._x*v2._y)-(v1._y*v2._x);
    return;

}