/////////////////////////////////////////////////////////////////////////////////////
/* PREFIX SCAN */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef PREFIX
#define PREFIX
#include "prefix_scan.h"
#endif

#ifndef TEST
#define TEST
#include "../Test/test.h"
#endif



__device__ void block_prescan(size_t *g_odata, size_t *g_idata, size_t *aux, const size_t size)
{

    __shared__ size_t temp[2*BLOCKSIZE];
    int thid = threadIdx.x;

    int offset = 1;

    temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
    temp[2 * thid + 1] = g_idata[2 * thid + 1];

    for (int d = size >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0)
    {
        aux[blockIdx.x] = temp[size - 1];
        temp[size - 1] = 0;
    } // clear the last element

    for (int d = 1; d < size; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            size_t t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // exclusive scan is turned into inclusive scan
    if (threadIdx.x != (blockDim.x - 1))
    {
        g_odata[2 * thid] = temp[2 * thid + 1]; // write results to device memory
        g_odata[2 * thid + 1] = temp[2 * thid + 2];
    }
    if (threadIdx.x == (blockDim.x - 1))
    {
        g_odata[2 * (blockDim.x - 1)] = temp[2 * (blockDim.x - 1) + 1]; // write results to device memory
        g_odata[2 * (blockDim.x - 1) + 1] = aux[blockIdx.x];
    }
    

}





void seq_prefix_inclusive(size_t* o_array, size_t* i_array, size_t array_size){

    o_array[0] = i_array[0];
    // Adding present element with previous element
    for (size_t i = 1; i < array_size; i++)
    {
        o_array[i] = o_array[i-1] + i_array[i];
    }
}


void master_prescan(size_t* o_array, size_t* i_array, size_t array_size, size_t array_bytes, int mode){

    /////////////////////////// 
    // device var
    int dev;
    
    ///////////////////////////
    // array var

    size_t array_grid_size;
    size_t array_rem_grid_size;
    size_t array_loop_cnt;	
    size_t array_fsize;
    size_t array_fbytes;

    ///////////////////////////
    // aux var
    size_t aux_size;
    size_t aux_grid_size;
    size_t aux_rem_grid_size;
    size_t aux_loop_cnt;
    size_t aux_fsize;
    size_t aux_fbytes;

    ///////////////////////////
    // aux_2 var
    size_t aux_2_size;
    size_t aux_2_grid_size;
    size_t aux_2_rem_grid_size;
    size_t aux_2_loop_cnt;
    size_t aux_2_fsize;
    size_t aux_2_fbytes;

    ///////////////////////////
    // aux_3 var
    size_t aux_3_size;
    size_t aux_3_grid_size;
    size_t aux_3_rem_grid_size;
    size_t aux_3_loop_cnt;
    size_t aux_3_fsize;
    size_t aux_3_fbytes;

    ///////////////////////////
    // aux_4 var
    size_t aux_4_grid_size;
    size_t aux_4_fsize;
    size_t aux_4_fbytes;

    ///////////////////////////
    // aux_5 var
    size_t aux_5_fsize;
    size_t aux_5_fbytes;

    ///////////////////////////
    // memory var
    size_t* d_data;
    size_t* o_data;
    size_t* t_data;

    size_t* aux_data;
    size_t* aux_2_data;
    size_t* aux_3_data;
    size_t* aux_4_data;
    size_t* aux_5_data;

    ///////////////////////////
    // device set up    
    dev = 0;
    CHECK(cudaSetDevice(dev));

    ///////////////////////////
    // array set up
    workload_calc(&array_grid_size, &array_rem_grid_size, &array_loop_cnt, &array_fsize, array_size);
    array_fbytes = array_fsize*sizeof(size_t);

    ///////////////////////////
    // aux set up
    aux_size = array_grid_size*array_loop_cnt + array_rem_grid_size;
    workload_calc(&aux_grid_size, &aux_rem_grid_size, &aux_loop_cnt, &aux_fsize, aux_size);
    aux_fbytes = aux_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_2 set up
    aux_2_size = aux_grid_size*aux_loop_cnt + aux_rem_grid_size;
    workload_calc(&aux_2_grid_size, &aux_2_rem_grid_size, &aux_2_loop_cnt, &aux_2_fsize, aux_2_size);
    aux_2_fbytes = aux_2_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_3 set up
    aux_3_size = aux_2_grid_size*aux_2_loop_cnt + aux_2_rem_grid_size;
    workload_calc(&aux_3_grid_size, &aux_3_rem_grid_size, &aux_3_loop_cnt, &aux_3_fsize, aux_3_size);
    aux_3_fbytes = aux_3_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_4 set up
    aux_4_grid_size = 1;
    aux_4_fsize = 2*BLOCKSIZE;
    aux_4_fbytes = aux_4_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_5 set up
    aux_5_fsize = 2*BLOCKSIZE;
    aux_5_fbytes = aux_5_fsize*sizeof(size_t);

    ///////////////////////////
    // memory set up
    #if MEMORY_MODEL == STD_MEMORY || MEMORY_MODEL == PINNED_MEMORY
        CHECK(cudaMalloc((size_t **)&d_data, array_fbytes));
    #endif
    CHECK(cudaMalloc((size_t **)&o_data, array_fbytes));
    CHECK(cudaMalloc((size_t **)&t_data, array_fbytes));


    CHECK(cudaMalloc((size_t **)&aux_data, aux_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_2_data, aux_2_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_3_data, aux_3_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_4_data, aux_4_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_5_data, aux_5_fbytes));
    


    // copy data and workloads from host to device

    #if MEMORY_MODEL == ZERO_MEMORY
        CHECK(cudaHostGetDevicePointer((void **)&d_data, (void *)i_array, 0));
    #else
        //CHECK(cudaMemset(d_data, 0, array_fbytes));
        CHECK(cudaMemcpy(d_data, i_array, array_bytes, cudaMemcpyHostToDevice));
    #endif
    ///////////////////////////
    // prefix scan

    // prefix on array
    for(int i = 0; i < array_loop_cnt; i++){
        split_prescan<<<array_grid_size, BLOCKSIZE>>>
        (t_data, d_data, aux_data, i);
	}
    if(array_rem_grid_size > 0){
        split_prescan<<<array_rem_grid_size, BLOCKSIZE>>>
        (t_data, d_data, aux_data, array_loop_cnt);
    }

    // prefix on aux
    for(int i = 0; i < aux_loop_cnt; i++){
        split_prescan<<<aux_grid_size, BLOCKSIZE>>>
        (aux_data, aux_data, aux_2_data, i);
    }
    if(aux_rem_grid_size > 0){
        split_prescan<<<aux_rem_grid_size, BLOCKSIZE>>>
        (aux_data, aux_data, aux_2_data, aux_loop_cnt);
    }

    // prefix on aux_2
    for(int i = 0; i < aux_2_loop_cnt; i++){
        split_prescan<<<aux_2_grid_size, BLOCKSIZE>>>
        (aux_2_data, aux_2_data, aux_3_data, i);
    }
    if(aux_2_rem_grid_size > 0){
        split_prescan<<<aux_2_rem_grid_size, BLOCKSIZE>>>
        (aux_2_data, aux_2_data, aux_3_data, aux_2_loop_cnt);
    }


    // prefix on aux_3
    for(int i = 0; i < aux_3_loop_cnt; i++){
        split_prescan<<<aux_3_grid_size, BLOCKSIZE>>>
        (aux_3_data, aux_3_data, aux_4_data, i);
    }
    if(aux_3_rem_grid_size > 0){
        split_prescan<<<aux_3_rem_grid_size, BLOCKSIZE>>>
        (aux_3_data, aux_3_data, aux_4_data, aux_3_loop_cnt);
    }

    
    // prefix on aux_4
    split_prescan<<<aux_4_grid_size, BLOCKSIZE>>>
    (aux_4_data, aux_4_data, aux_5_data, 0);

    
    // aux_3+aux_4
    for(int i = 0; i < aux_3_loop_cnt; i++){
        add_block<<<aux_3_grid_size, BLOCKSIZE>>>
        (aux_3_data, aux_4_data, i);
    }
    if(aux_3_rem_grid_size > 0){
        add_block<<<aux_3_rem_grid_size, BLOCKSIZE>>>
        (aux_3_data, aux_4_data, aux_3_loop_cnt);       
    }

    
    // aux_2+aux_3
    for(int i = 0; i < aux_2_loop_cnt; i++){
        add_block<<<aux_2_grid_size, BLOCKSIZE>>>
        (aux_2_data, aux_3_data, i);
    }
    if(aux_2_rem_grid_size > 0){
        add_block<<<aux_2_rem_grid_size, BLOCKSIZE>>>
        (aux_2_data, aux_3_data, aux_2_loop_cnt);       
    }

    // aux+aux_2
    for(int i = 0; i < aux_loop_cnt; i++){
        add_block<<<aux_grid_size, BLOCKSIZE>>>
        (aux_data, aux_2_data, i);
    }
    if(aux_rem_grid_size > 0){
        add_block<<<aux_rem_grid_size, BLOCKSIZE>>>
        (aux_data, aux_2_data, aux_loop_cnt);       
    }

    // array+aux
    for(int i = 0; i < array_loop_cnt; i++){
        add_block<<<array_grid_size, BLOCKSIZE>>>
        (t_data, aux_data, i);
    }
    if(array_rem_grid_size > 0){
        add_block<<<array_rem_grid_size, BLOCKSIZE>>>
        (t_data, aux_data, array_loop_cnt);      
    }


    if(mode == EXCLUSIVE){
        // array gets shifted by one for exclusive sum
        for(int i = 0; i < array_loop_cnt; i++){
            shift_block<<<array_grid_size, BLOCKSIZE>>>
            (o_data, t_data, array_fsize, i);
        }
        if(array_rem_grid_size > 0){
            shift_block<<<array_rem_grid_size, BLOCKSIZE>>>
            (o_data, t_data, array_fsize, array_loop_cnt); 
        }
    }
    else{
        size_t* temp;
        temp = o_data;
        o_data = t_data;
        t_data = temp;
    }

    // copy back results
    CHECK(cudaMemcpy(o_array, o_data, array_bytes, cudaMemcpyDeviceToHost));

    // free memory
    #if MEMORY_MODEL == STD_MEMORY || MEMORY_MODEL == PINNED_MEMORY
        CHECK(cudaFree(d_data));
    #endif
    CHECK(cudaFree(t_data));
    CHECK(cudaFree(o_data));


    CHECK(cudaFree(aux_data));
    CHECK(cudaFree(aux_2_data));
    CHECK(cudaFree(aux_3_data));
    CHECK(cudaFree(aux_4_data));


}


void master_prescan_gpu(size_t* o_array, size_t* i_array, size_t array_fsize, size_t array_fbytes, 
                            size_t array_grid_size, size_t array_rem_grid_size, size_t array_loop_cnt, int mode){


    ///////////////////////////
    // aux var
    size_t aux_size;
    size_t aux_grid_size;
    size_t aux_rem_grid_size;
    size_t aux_loop_cnt;
    size_t aux_fsize;
    size_t aux_fbytes;

    ///////////////////////////
    // aux_2 var
    size_t aux_2_size;
    size_t aux_2_grid_size;
    size_t aux_2_rem_grid_size;
    size_t aux_2_loop_cnt;
    size_t aux_2_fsize;
    size_t aux_2_fbytes;

    ///////////////////////////
    // aux_3 var
    size_t aux_3_size;
    size_t aux_3_grid_size;
    size_t aux_3_rem_grid_size;
    size_t aux_3_loop_cnt;
    size_t aux_3_fsize;
    size_t aux_3_fbytes;

    ///////////////////////////
    // aux_4 var
    size_t aux_4_grid_size;
    size_t aux_4_fsize;
    size_t aux_4_fbytes;

    ///////////////////////////
    // aux_5 var
    size_t aux_5_fsize;
    size_t aux_5_fbytes;

    ///////////////////////////
    // memory var
    size_t* t_data;

    size_t* aux_data;
    size_t* aux_2_data;
    size_t* aux_3_data;
    size_t* aux_4_data;
    size_t* aux_5_data;



    ///////////////////////////
    // aux set up
    aux_size = array_grid_size*array_loop_cnt + array_rem_grid_size;
    workload_calc(&aux_grid_size, &aux_rem_grid_size, &aux_loop_cnt, &aux_fsize, aux_size);
    aux_fbytes = aux_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_2 set up
    aux_2_size = aux_grid_size*aux_loop_cnt + aux_rem_grid_size;
    workload_calc(&aux_2_grid_size, &aux_2_rem_grid_size, &aux_2_loop_cnt, &aux_2_fsize, aux_2_size);
    aux_2_fbytes = aux_2_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_3 set up
    aux_3_size = aux_2_grid_size*aux_2_loop_cnt + aux_2_rem_grid_size;
    workload_calc(&aux_3_grid_size, &aux_3_rem_grid_size, &aux_3_loop_cnt, &aux_3_fsize, aux_3_size);
    aux_3_fbytes = aux_3_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_4 set up
    aux_4_grid_size = 1;
    aux_4_fsize = 2*BLOCKSIZE;
    aux_4_fbytes = aux_4_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_5 set up
    aux_5_fsize = 2*BLOCKSIZE;
    aux_5_fbytes = aux_5_fsize*sizeof(size_t);

    ///////////////////////////
    // memory set up
    CHECK(cudaMalloc((size_t **)&t_data, array_fbytes));

    CHECK(cudaMalloc((size_t **)&aux_data, aux_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_2_data, aux_2_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_3_data, aux_3_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_4_data, aux_4_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_5_data, aux_5_fbytes));

    ///////////////////////////
    // prefix scan

    // prefix on array
    for(size_t i = 0; i < array_loop_cnt; i++){
        split_prescan<<<array_grid_size, BLOCKSIZE>>>
        (t_data, i_array, aux_data, i);
	}
    if(array_rem_grid_size > 0){
        split_prescan<<<array_rem_grid_size, BLOCKSIZE>>>
        (t_data, i_array, aux_data, array_loop_cnt);
    }

    // prefix on aux
    for(int i = 0; i < aux_loop_cnt; i++){
        split_prescan<<<aux_grid_size, BLOCKSIZE>>>
        (aux_data, aux_data, aux_2_data, i);
    }
    if(aux_rem_grid_size > 0){
        split_prescan<<<aux_rem_grid_size, BLOCKSIZE>>>
        (aux_data, aux_data, aux_2_data, aux_loop_cnt);
    }

    // prefix on aux_2
    for(int i = 0; i < aux_2_loop_cnt; i++){
        split_prescan<<<aux_2_grid_size, BLOCKSIZE>>>
        (aux_2_data, aux_2_data, aux_3_data, i);
    }
    if(aux_2_rem_grid_size > 0){
        split_prescan<<<aux_2_rem_grid_size, BLOCKSIZE>>>
        (aux_2_data, aux_2_data, aux_3_data, aux_2_loop_cnt);
    }


    // prefix on aux_3
    for(int i = 0; i < aux_3_loop_cnt; i++){
        split_prescan<<<aux_3_grid_size, BLOCKSIZE>>>
        (aux_3_data, aux_3_data, aux_4_data, i);
    }
    if(aux_3_rem_grid_size > 0){
        split_prescan<<<aux_3_rem_grid_size, BLOCKSIZE>>>
        (aux_3_data, aux_3_data, aux_4_data, aux_3_loop_cnt);
    }

    
    // prefix on aux_4
    split_prescan<<<aux_4_grid_size, BLOCKSIZE>>>
    (aux_4_data, aux_4_data, aux_5_data, 0);

    
    // aux_3+aux_4
    for(int i = 0; i < aux_3_loop_cnt; i++){
        add_block<<<aux_3_grid_size, BLOCKSIZE>>>
        (aux_3_data, aux_4_data, i);
    }
    if(aux_3_rem_grid_size > 0){
        add_block<<<aux_3_rem_grid_size, BLOCKSIZE>>>
        (aux_3_data, aux_4_data, aux_3_loop_cnt);      
    }

    
    // aux_2+aux_3
    for(int i = 0; i < aux_2_loop_cnt; i++){
        add_block<<<aux_2_grid_size, BLOCKSIZE>>>
        (aux_2_data, aux_3_data, i);
    }
    if(aux_2_rem_grid_size > 0){
        add_block<<<aux_2_rem_grid_size, BLOCKSIZE>>>
        (aux_2_data, aux_3_data, aux_2_loop_cnt);       
    }

    // aux+aux_2
    for(int i = 0; i < aux_loop_cnt; i++){
        add_block<<<aux_grid_size, BLOCKSIZE>>>
        (aux_data, aux_2_data, i);
    }
    if(aux_rem_grid_size > 0){
        add_block<<<aux_rem_grid_size, BLOCKSIZE>>>
        (aux_data, aux_2_data, aux_loop_cnt);      
    }

    // array+aux
    for(int i = 0; i < array_loop_cnt; i++){
        add_block<<<array_grid_size, BLOCKSIZE>>>
        (t_data, aux_data, i);
    }
    if(array_rem_grid_size > 0){
        add_block<<<array_rem_grid_size, BLOCKSIZE>>>
        (t_data, aux_data, array_loop_cnt);   
    }

    if(mode == EXCLUSIVE){
        // array gets shifted by one for exclusive sum
        for(int i = 0; i < array_loop_cnt; i++){
            shift_block<<<array_grid_size, BLOCKSIZE>>>
            (o_array, t_data, array_fsize, i);
        }
        if(array_rem_grid_size > 0){
            shift_block<<<array_rem_grid_size, BLOCKSIZE>>>
            (o_array, t_data, array_fsize, array_loop_cnt); 
        }
    }
    else{
        size_t* temp;
        temp = o_array;
        o_array = t_data;
        t_data = temp;
    }
    
    // free memory
    CHECK(cudaFree(t_data));


    CHECK(cudaFree(aux_data));
    CHECK(cudaFree(aux_2_data));
    CHECK(cudaFree(aux_3_data));
    CHECK(cudaFree(aux_4_data));
    CHECK(cudaFree(aux_5_data));


}




void master_stream_prescan_gpu(size_t* o_array, size_t* i_array, size_t array_fsize, size_t array_fbytes, 
                            int array_grid_size, int array_rem_grid_size, int array_loop_cnt, int mode, cudaStream_t stream){

    ///////////////////////////
    // aux var
    size_t aux_size;
    size_t aux_grid_size;
    size_t aux_rem_grid_size;
    size_t aux_loop_cnt;
    size_t aux_fsize;
    size_t aux_fbytes;

    ///////////////////////////
    // aux_2 var
    size_t aux_2_size;
    size_t aux_2_grid_size;
    size_t aux_2_rem_grid_size;
    size_t aux_2_loop_cnt;
    size_t aux_2_fsize;
    size_t aux_2_fbytes;

    ///////////////////////////
    // aux_3 var
    size_t aux_3_size;
    size_t aux_3_grid_size;
    size_t aux_3_rem_grid_size;
    size_t aux_3_loop_cnt;
    size_t aux_3_fsize;
    size_t aux_3_fbytes;

    ///////////////////////////
    // aux_4 var
    size_t aux_4_grid_size;
    size_t aux_4_fsize;
    size_t aux_4_fbytes;

    ///////////////////////////
    // aux_5 var
    size_t aux_5_fsize;
    size_t aux_5_fbytes;

    ///////////////////////////
    // memory var
    size_t* t_data;

    size_t* aux_data;
    size_t* aux_2_data;
    size_t* aux_3_data;
    size_t* aux_4_data;
    size_t* aux_5_data;

    ///////////////////////////
    // aux set up
    aux_size = array_grid_size*array_loop_cnt + array_rem_grid_size;
    workload_calc(&aux_grid_size, &aux_rem_grid_size, &aux_loop_cnt, &aux_fsize, aux_size);
    aux_fbytes = aux_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_2 set up
    aux_2_size = aux_grid_size*aux_loop_cnt + aux_rem_grid_size;
    workload_calc(&aux_2_grid_size, &aux_2_rem_grid_size, &aux_2_loop_cnt, &aux_2_fsize, aux_2_size);
    aux_2_fbytes = aux_2_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_3 set up
    aux_3_size = aux_2_grid_size*aux_2_loop_cnt + aux_2_rem_grid_size;
    workload_calc(&aux_3_grid_size, &aux_3_rem_grid_size, &aux_3_loop_cnt, &aux_3_fsize, aux_3_size);
    aux_3_fbytes = aux_3_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_4 set up
    aux_4_grid_size = 1;
    aux_4_fsize = 2*BLOCKSIZE;
    aux_4_fbytes = aux_4_fsize*sizeof(size_t);

    ///////////////////////////
    // aux_5 set up
    aux_5_fsize = 2*BLOCKSIZE;
    aux_5_fbytes = aux_5_fsize*sizeof(size_t);

    ///////////////////////////
    // memory set up
    CHECK(cudaMalloc((size_t **)&t_data, array_fbytes));


    CHECK(cudaMalloc((size_t **)&aux_data, aux_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_2_data, aux_2_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_3_data, aux_3_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_4_data, aux_4_fbytes));
    CHECK(cudaMalloc((size_t **)&aux_5_data, aux_5_fbytes));

    

    ///////////////////////////
    // prefix scan

    // prefix on array
    for(size_t i = 0; i < array_loop_cnt; i++){
        split_prescan<<<array_grid_size, BLOCKSIZE, 0, stream>>>
        (t_data, i_array, aux_data, i);
	}
    if(array_rem_grid_size > 0){
        split_prescan<<<array_rem_grid_size, BLOCKSIZE, 0, stream>>>
        (t_data, i_array, aux_data, array_loop_cnt);
    }


    // prefix on aux
    for(int i = 0; i < aux_loop_cnt; i++){
        split_prescan<<<aux_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_data, aux_data, aux_2_data, i);
    }
    if(aux_rem_grid_size > 0){
        split_prescan<<<aux_rem_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_data, aux_data, aux_2_data, aux_loop_cnt);
    }


    // prefix on aux_2
    for(int i = 0; i < aux_2_loop_cnt; i++){
        split_prescan<<<aux_2_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_2_data, aux_2_data, aux_3_data, i);
    }
    if(aux_2_rem_grid_size > 0){
        split_prescan<<<aux_2_rem_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_2_data, aux_2_data, aux_3_data, aux_2_loop_cnt);
    }


    // prefix on aux_3
    for(int i = 0; i < aux_3_loop_cnt; i++){
        split_prescan<<<aux_3_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_3_data, aux_3_data, aux_4_data, i);
    }
    if(aux_3_rem_grid_size > 0){
        split_prescan<<<aux_3_rem_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_3_data, aux_3_data, aux_4_data, aux_3_loop_cnt);
    }


    // prefix on aux_4
    split_prescan<<<aux_4_grid_size, BLOCKSIZE, 0, stream>>>
    (aux_4_data, aux_4_data, aux_5_data, 0);


    // aux_3+aux_4
    for(int i = 0; i < aux_3_loop_cnt; i++){
        add_block<<<aux_3_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_3_data, aux_4_data, i);
    }
    if(aux_3_rem_grid_size > 0){
        add_block<<<aux_3_rem_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_3_data, aux_4_data, aux_3_loop_cnt);       
    }


    
    // aux_2+aux_3
    for(int i = 0; i < aux_2_loop_cnt; i++){
        add_block<<<aux_2_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_2_data, aux_3_data, i);
    }
    if(aux_2_rem_grid_size > 0){
        add_block<<<aux_2_rem_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_2_data, aux_3_data, aux_2_loop_cnt);       
    }


    // aux+aux_2
    for(int i = 0; i < aux_loop_cnt; i++){
        add_block<<<aux_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_data, aux_2_data, i);
    }
    if(aux_rem_grid_size > 0){
        add_block<<<aux_rem_grid_size, BLOCKSIZE, 0, stream>>>
        (aux_data, aux_2_data, aux_loop_cnt);       
    }


    // array+aux
    for(int i = 0; i < array_loop_cnt; i++){
        add_block<<<array_grid_size, BLOCKSIZE, 0, stream>>>
        (t_data, aux_data, i);
    }
    if(array_rem_grid_size > 0){
        add_block<<<array_rem_grid_size, BLOCKSIZE, 0, stream>>>
        (t_data, aux_data, array_loop_cnt);      
    }

    

    if(mode == EXCLUSIVE){
        // array gets shifted by one for exclusive sum
        for(int i = 0; i < array_loop_cnt; i++){
            shift_block<<<array_grid_size, BLOCKSIZE, 0, stream>>>
            (o_array, t_data, array_fsize, i);
        }
        if(array_rem_grid_size > 0){
            shift_block<<<array_rem_grid_size, BLOCKSIZE, 0, stream>>>
            (o_array, t_data, array_fsize, array_loop_cnt); 
        }

    }
    else{
        size_t* temp;
        temp = o_array;
        o_array = t_data;
        t_data = temp;
    }

    // free memory
    CHECK(cudaFree(t_data));


    CHECK(cudaFree(aux_data));
    CHECK(cudaFree(aux_2_data));
    CHECK(cudaFree(aux_3_data));
    CHECK(cudaFree(aux_4_data));


}

__global__ void split_prescan(size_t* g_odata, size_t* g_idata, size_t* aux, int block_offset)
{
    
    block_prescan(
    &g_odata[blockIdx.x * 2 * blockDim.x+block_offset*MAX_BLOCK_COUNT_SHIFT], 
    &g_idata[blockIdx.x * 2 * blockDim.x+block_offset*MAX_BLOCK_COUNT_SHIFT], 
    &aux[block_offset*MAX_BLOCK_COUNT], 
    2 * blockDim.x);
    
}



__global__ void shift_block(size_t* o_data, size_t* t_data, size_t array_size, int block_offset)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;


    int index_1 = 2 * thid + block_offset*MAX_BLOCK_COUNT_SHIFT;
    int index_2 = index_1 + 1;

    if (index_1 == 0)
    {
        o_data[0] = 0;
        o_data[index_1 + 1] = t_data[index_1];
        o_data[index_2 + 1] = t_data[index_2];
    }
    else if (index_1 == array_size-2) 
    {
        o_data[index_1 + 1] = t_data[index_1];
    }
    else if (index_1 < array_size-2)
    {
        o_data[index_1 + 1] = t_data[index_1];
        o_data[index_2 + 1] = t_data[index_2];
    }
    
}

__global__ void add_block(size_t* o_data, size_t* aux, int block_offset)
{
    int thid = (blockIdx.x) * blockDim.x + threadIdx.x;

    if(blockIdx.x == 0 && block_offset == 0){
        return;
    }

    int index_1 = 2 * thid + block_offset*MAX_BLOCK_COUNT_SHIFT;
    int index_2 = index_1 + 1;


    o_data[index_1] += aux[blockIdx.x-1+block_offset*MAX_BLOCK_COUNT];
    o_data[index_2] += aux[blockIdx.x-1+block_offset*MAX_BLOCK_COUNT];


        

    
}