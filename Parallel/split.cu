/////////////////////////////////////////////////////////////////////////////////////
/* SPLIT */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef SPLIT
#define SPLIT
#include "split.h"
#endif


void split_point_array(Point_array_par* points, Point_array_par* points_above, 
    Point_array_par* points_below, Line* l){

    // workload array var
    size_t array_grid_size;
    size_t array_rem_grid_size;
    size_t array_loop_cnt;
    size_t array_fsize;
    size_t array_fbytes;
    size_t point_bytes;
    size_t point_fbytes;


    // memory var
    Point* points_gpu;
    Point* temp_above;
    Point* temp_below;  
    size_t* above_bits;
    size_t* below_bits;
    size_t* above_index;
    size_t* below_index;

    // workload calc
    workload_calc(&array_grid_size, &array_rem_grid_size, &array_loop_cnt, &array_fsize, points->size);
    point_fbytes = array_fsize*sizeof(Point);
    array_fbytes = array_fsize*sizeof(size_t);
    point_bytes = points->size*sizeof(Point);

    // set up memory
    CHECK(cudaMalloc((Point **)&points_gpu, point_fbytes));
    CHECK(cudaMalloc((size_t **)&above_bits, array_fbytes));
    CHECK(cudaMalloc((size_t **)&below_bits, array_fbytes));
    CHECK(cudaMalloc((size_t **)&above_index, array_fbytes+sizeof(size_t)));
    CHECK(cudaMalloc((size_t **)&below_index, array_fbytes+sizeof(size_t)));

    // transfer point array and workload
    #if MEMORY_MODEL == ZERO_MEMORY
        CHECK(cudaHostGetDevicePointer((void **)&points_gpu, (void *)points->array, 0));
    #else
        CHECK(cudaMemcpy(points_gpu, points->array, point_bytes, cudaMemcpyHostToDevice));
    #endif

    // set bits in above/below array to 1/0 // 0/1 // 0/0
    for(int i = 0; i < array_loop_cnt; i++){
        setbits<<<array_grid_size, BLOCKSIZE>>>(above_bits, below_bits, points_gpu, l, points->size, i);
    }
    if(array_rem_grid_size > 0){
        setbits<<<array_grid_size, BLOCKSIZE>>>(above_bits, below_bits, points_gpu, l, points->size, array_loop_cnt);  
    }

    // prefix bits to get indexes
    master_prescan_gpu(above_index, above_bits, array_fsize, array_fbytes, 
                        array_grid_size, array_rem_grid_size, array_loop_cnt, EXCLUSIVE);

    master_prescan_gpu(below_index, below_bits, array_fsize, array_fbytes, 
                        array_grid_size, array_rem_grid_size, array_loop_cnt, EXCLUSIVE);

    
    // set up memory for output point arrays
    
    CHECK(cudaMalloc((Point **)&temp_above, point_fbytes));
    CHECK(cudaMalloc((Point **)&temp_below, point_fbytes));

    // move values
    for(int i = 0; i < array_loop_cnt; i++){
        movevalues<<<array_grid_size, BLOCKSIZE>>>(temp_above, points_gpu, above_bits, 
                                                above_index, array_fsize, i);
    }
    if(array_rem_grid_size > 0){
        movevalues<<<array_grid_size, BLOCKSIZE>>>(temp_above, points_gpu, above_bits, 
                                                above_index, array_fsize, array_loop_cnt);
    }
    // copy size and values back
    points_above->array = temp_above;
    CHECK(cudaMemcpy(&points_above->size, above_index+array_fsize, sizeof(size_t), cudaMemcpyDeviceToHost));



    // move values
    for(int i = 0; i < array_loop_cnt; i++){
        movevalues<<<array_grid_size, BLOCKSIZE>>>(temp_below, points_gpu, below_bits, 
                                                below_index, array_fsize, i);
    }
    if(array_rem_grid_size > 0){
         movevalues<<<array_grid_size, BLOCKSIZE>>>(temp_below, points_gpu, below_bits, 
                                                below_index, array_fsize, array_loop_cnt);       
    }
    // copy size and values back
    points_below->array = temp_below;
    CHECK(cudaMemcpy(&points_below->size, below_index+array_fsize, sizeof(size_t), cudaMemcpyDeviceToHost));



    // free memory
    #if MEMORY_MODEL == STD_MEMORY || MEMORY_MODEL == PINNED_MEMORY
        CHECK(cudaFree(points_gpu));
    #endif
    CHECK(cudaFree(above_bits));
    CHECK(cudaFree(below_bits));
    CHECK(cudaFree(above_index));
    CHECK(cudaFree(below_index));

}


void split_point_array_side(Point_array_par* points, Point_array_par* points_side, Line* l, int side){

    // workload array var
    size_t array_grid_size;
    size_t array_rem_grid_size;
    size_t array_loop_cnt;
    size_t array_fsize;
    size_t array_fbytes;
    size_t point_fbytes;

    // memory var
    Point* points_gpu;
    Point* temp_side;
    size_t* side_bits;
    size_t* side_index;

    // workload calc
    workload_calc(&array_grid_size, &array_rem_grid_size, &array_loop_cnt, &array_fsize, points->size);
    point_fbytes = array_fsize*sizeof(Point);
    array_fbytes = array_fsize*sizeof(size_t);

    // set up memory
    CHECK(cudaMalloc((size_t **)&side_bits, array_fbytes+sizeof(size_t)));
    CHECK(cudaMalloc((size_t **)&side_index, array_fbytes+sizeof(size_t)));

    // set bits in above/below array to 1/0 // 0/1 // 0/0
    for(int i = 0; i < array_loop_cnt; i++){
        setbits_side<<<array_grid_size, BLOCKSIZE>>>(side_bits, points_gpu, l, points->size, side, i);
    }
    if(array_rem_grid_size > 0){
        setbits_side<<<array_grid_size, BLOCKSIZE>>>(side_bits, points_gpu, l, points->size, side, array_loop_cnt);
    }
    

    // prefix bits to get indexes
    master_prescan_gpu(side_index, side_bits, array_fsize, array_fbytes, 
                        array_grid_size, array_rem_grid_size, array_loop_cnt, EXCLUSIVE);
                        

    // set up memory for output point arrays
    CHECK(cudaMalloc((Point **)&temp_side, point_fbytes));


    // move values
    for(int i = 0; i < array_loop_cnt; i++){
    movevalues<<<array_grid_size, BLOCKSIZE>>>(temp_side, points_gpu, side_bits, 
                                                side_index, array_fsize, i);
    }
    if(array_rem_grid_size > 0){
        movevalues<<<array_grid_size, BLOCKSIZE>>>(temp_side, points_gpu, side_bits, 
                                                side_index, array_fsize, array_loop_cnt);
    }

    points_side->array = temp_side;
    CHECK(cudaMemcpy(&points_side->size, side_index+array_fsize, sizeof(size_t), cudaMemcpyDeviceToHost));

    // free memory
    CHECK(cudaFree(points_gpu));
    CHECK(cudaFree(side_bits));
    CHECK(cudaFree(side_index));

}


__global__ void setbits(size_t *above_bits, size_t *below_bits, Point* points, Line* l, 
                            size_t array_size, int block_offset)
{
    int thid = (blockIdx.x) * blockDim.x + threadIdx.x;

    int index_1 = 2 * thid + block_offset*MAX_BLOCK_COUNT_SHIFT;
    int index_2 = index_1 + 1;
    int result;

    if(index_1 < array_size){
        
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
    }
    else{
        above_bits[index_1] = 0;
        below_bits[index_1] = 0;
    }

    if(index_2 < array_size){

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
    else{
        above_bits[index_2] = 0;
        below_bits[index_2] = 0;
    }
        
    

}

__global__ void setbits_side(size_t *bits, Point* points, Line* l, 
                            size_t array_size, int side, int block_offset)
{
    int thid = (blockIdx.x) * blockDim.x + threadIdx.x;

    int index_1 = 2 * thid + block_offset*MAX_BLOCK_COUNT_SHIFT;
    int index_2 = index_1 + 1;
    int result;

    if(index_1 < array_size){
        check_point_location_gpu(l, points[index_1], &result);

        if(result == side){
            bits[index_1] = 1;
        }
        else{
            bits[index_1] = 0;
        }
    }
    else{
        bits[index_1] = 0;
    }

    if(index_2 < array_size){
        check_point_location_gpu(l, points[index_2], &result);

        if(result == side){
            bits[index_2] = 1;
        }
        else{
            bits[index_2] = 0;
        }
    }
    else{
        bits[index_1] = 0;
    }
        
    


}



__global__ void movevalues(Point* o_data, Point* i_data, size_t* bit_data,
                           size_t* index_data, size_t array_fsize, int block_offset)
{

    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if(thid == 0){
        index_data[array_fsize] = bit_data[array_fsize-1]+index_data[array_fsize-1];
    }


        
    int index_1 = 2 * thid + block_offset*MAX_BLOCK_COUNT_SHIFT;
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


__device__ void check_point_location_gpu(Line* l, Point z, int* result){

    Vector v1;
    Vector v2; 
    
    double cross_result;

    init_vector_gpu(l->p, l->q, &v1);
    init_vector_gpu(l->p, z, &v2);
    
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