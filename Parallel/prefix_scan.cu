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



__device__ void block_prescan(unsigned long long int *g_odata, unsigned long long int *g_idata, unsigned long long int *aux, unsigned long long int *temp, const size_t size)
{

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
            unsigned long long int t = temp[ai];
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



__global__ void split_prescan(unsigned long long int *g_odata, unsigned long long int *g_idata, unsigned long long int *aux, int* workload)
{
    extern __shared__ unsigned long long int temp[];
    
    for(int i = 0; i<workload[blockIdx.x]; i++){
        block_prescan(&g_odata[blockIdx.x * 2 * blockDim.x+i*MAX_BLOCK_COUNT_SHIFT], &g_idata[blockIdx.x * 2 * blockDim.x+i*MAX_BLOCK_COUNT_SHIFT], &aux[i*MAX_BLOCK_COUNT], temp, 2 * blockDim.x);
    }
}



__global__ void shift_block(unsigned long long int *o_data, unsigned long long int *t_data, size_t array_size, int* workload)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < workload[blockIdx.x]; i++){
        int index_1 = 2 * thid + i*MAX_BLOCK_COUNT_SHIFT;
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
}

__global__ void add_block(unsigned long long int *o_data, unsigned long long int *aux, size_t array_size, int* workload)
{
    int thid = (blockIdx.x) * blockDim.x + threadIdx.x;

    for(int i = 0; i < workload[blockIdx.x]; i++){

        if(blockIdx.x == 0 && i == 0){
            continue;
        }

        int index_1 = 2 * thid + i*MAX_BLOCK_COUNT_SHIFT;
        int index_2 = index_1 + 1;

        if(index_1 < array_size){

            if(index_1 == array_size-1){
                o_data[index_1] += aux[blockIdx.x-1+i*MAX_BLOCK_COUNT];
            }
            else{
                o_data[index_1] += aux[blockIdx.x-1+i*MAX_BLOCK_COUNT];
                o_data[index_2] += aux[blockIdx.x-1+i*MAX_BLOCK_COUNT];
            }
        }
        

    }
}

void seq_prefix_inclusive(unsigned long long int* o_array, unsigned long long int* i_array, size_t array_size){

    o_array[0] = i_array[0];
    // Adding present element with previous element
    for (size_t i = 1; i < array_size; i++)
    {
        o_array[i] = o_array[i-1] + i_array[i];
    }
}


void master_prescan(unsigned long long int* o_array, unsigned long long int* i_array, size_t array_size, size_t array_bytes, int mode){

    /////////////////////////// 
    // device var
    int dev;
    cudaDeviceProp deviceProp;
    size_t shared_mem_size;
    size_t shared_mem_bytes;
    
    ///////////////////////////
    // array var

    int array_grid_size;
    int* array_workload;

    size_t array_fsize;
    size_t array_fbytes;

    ///////////////////////////
    // aux var
    size_t aux_size;
    
    int aux_grid_size;
    int* aux_workload;

    size_t aux_fsize;
    size_t aux_fbytes;

    ///////////////////////////
    // aux_2 var
    size_t aux_2_size;
    
    int aux_2_grid_size;
    int* aux_2_workload;

    size_t aux_2_fsize;
    size_t aux_2_fbytes;

    ///////////////////////////
    // aux_3 var
    size_t aux_3_fbytes;
    size_t aux_3_size;


    ///////////////////////////
    // memory var
    unsigned long long int* d_data;
    unsigned long long int* o_data;
    unsigned long long int* t_data;

    unsigned long long int* iaux_data;
    unsigned long long int* oaux_data;

    unsigned long long int* iaux_2_data;
    unsigned long long int* oaux_2_data;

    unsigned long long int* iaux_3_data;
    unsigned long long int* oaux_3_data;
    unsigned long long int* iaux_3_data_gpu;
    unsigned long long int* oaux_3_data_gpu;

    int* array_workload_gpu;
    int* aux_workload_gpu;
    int* aux_2_workload_gpu;

    ///////////////////////////
    // device set up

    dev = 0;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Starting reduction at ");
    printf("device %d: %s", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    
    shared_mem_size = 2*BLOCKSIZE; 
    shared_mem_bytes = shared_mem_size*sizeof(unsigned long long int);

    ///////////////////////////
    // array set up

    array_workload = workload_calc(&array_grid_size, &array_fsize, array_size);
    array_fbytes = array_fsize*sizeof(unsigned long long int);

    ///////////////////////////
    // aux set up
    aux_size = array_grid_size*array_workload[0];

    aux_workload = workload_calc(&aux_grid_size, &aux_fsize, aux_size);
    aux_fbytes = aux_fsize*sizeof(unsigned long long int);

    ///////////////////////////
    // aux_2 set up
    aux_2_size = aux_grid_size*aux_workload[0];

    aux_2_workload = workload_calc(&aux_2_grid_size, &aux_2_fsize, aux_2_size);
    aux_2_fbytes = aux_2_fsize*sizeof(unsigned long long int);

    ///////////////////////////
    // aux_3 set up
    aux_3_size = aux_2_fsize/(2*BLOCKSIZE);
    aux_3_fbytes = aux_3_size*sizeof(unsigned long long int);

    ///////////////////////////
    // memory set up
    CHECK(cudaMalloc((unsigned long long int **)&d_data, array_fbytes));
    CHECK(cudaMalloc((unsigned long long int **)&o_data, array_fbytes));
    CHECK(cudaMalloc((unsigned long long int **)&t_data, array_fbytes));

    CHECK(cudaMalloc((unsigned long long int **)&iaux_data, aux_fbytes));
    CHECK(cudaMalloc((unsigned long long int **)&oaux_data, aux_fbytes));

    CHECK(cudaMalloc((unsigned long long int **)&iaux_2_data, aux_2_fbytes));
    CHECK(cudaMalloc((unsigned long long int **)&oaux_2_data, aux_2_fbytes));

    CHECK(cudaMalloc((unsigned long long int **)&iaux_3_data_gpu, aux_3_fbytes));
    CHECK(cudaMalloc((unsigned long long int **)&oaux_3_data_gpu, aux_3_fbytes));
    iaux_3_data = (unsigned long long int*)malloc(aux_3_fbytes);
    oaux_3_data = (unsigned long long int*)malloc(aux_3_fbytes);
    if(!iaux_3_data || !oaux_3_data){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    CHECK(cudaMalloc((int **)&array_workload_gpu, array_grid_size*sizeof(int)));
    CHECK(cudaMalloc((int **)&aux_workload_gpu, aux_grid_size*sizeof(int)));
    CHECK(cudaMalloc((int **)&aux_2_workload_gpu, aux_2_grid_size*sizeof(int)));

    // copy data and workloads from host to device
    CHECK(cudaMemset(d_data, 0, array_fbytes));
    CHECK(cudaMemcpy(d_data, i_array, array_bytes, cudaMemcpyHostToDevice));

    CHECK(cudaMemset(iaux_data, 0, aux_fbytes));
    CHECK(cudaMemset(iaux_2_data, 0, aux_2_fbytes));

    CHECK(cudaMemcpy(array_workload_gpu, array_workload, array_grid_size*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(aux_workload_gpu, aux_workload, aux_grid_size*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(aux_2_workload_gpu, aux_2_workload, aux_2_grid_size*sizeof(int), cudaMemcpyHostToDevice));

    ///////////////////////////
    // prefix scan

    // prefix on array
    split_prescan<<<array_grid_size, BLOCKSIZE, shared_mem_bytes>>>
    (t_data, d_data, iaux_data, array_workload_gpu);

    // prefix on aux
    split_prescan<<<aux_grid_size, BLOCKSIZE, shared_mem_bytes>>>
    (oaux_data, iaux_data, iaux_2_data, aux_workload_gpu);

    // prefix on aux_2
    split_prescan<<<aux_2_grid_size, BLOCKSIZE, shared_mem_bytes>>>
    (oaux_2_data, iaux_2_data, iaux_3_data_gpu, aux_2_workload_gpu);

    // prefix on aux_3 locally sequentially
    CHECK(cudaMemcpy(iaux_3_data, iaux_3_data_gpu, aux_3_fbytes, cudaMemcpyDeviceToHost));
    seq_prefix_inclusive(oaux_3_data, iaux_3_data, aux_3_size);
    CHECK(cudaMemcpy(oaux_3_data_gpu, oaux_3_data, aux_3_fbytes, cudaMemcpyHostToDevice));

    // aux_2+aux_3
    add_block<<<aux_2_grid_size, BLOCKSIZE>>>
    (oaux_2_data, oaux_3_data_gpu, aux_2_fsize, aux_2_workload_gpu);

    // aux+aux_2
    add_block<<<aux_grid_size, BLOCKSIZE>>>
    (oaux_data, oaux_2_data, aux_fsize, aux_workload_gpu);

    // array+aux
    add_block<<<array_grid_size, BLOCKSIZE>>>
    (t_data, oaux_data, array_fsize, array_workload_gpu);

    if(mode == EXCLUSIVE){
        printf("hallo");
        // array gets shifted by one for exclusive sum
        shift_block<<<array_grid_size, BLOCKSIZE>>>
        (o_data, t_data, array_fsize, array_workload_gpu);
    }
    else{
        unsigned long long int * temp;
        temp = o_data;
        o_data = t_data;
        t_data = temp;
    }

    // copy back results
    CHECK(cudaMemcpy(o_array, o_data, array_bytes, cudaMemcpyDeviceToHost));

    // free memory
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(t_data));
    CHECK(cudaFree(o_data));
    CHECK(cudaFree(array_workload_gpu));
    free(array_workload);

    if(iaux_data){
        CHECK(cudaFree(iaux_data));
        CHECK(cudaFree(oaux_data));
        CHECK(cudaFree(aux_workload_gpu));
        free(aux_workload);
    }

    if(iaux_2_data){
        CHECK(cudaFree(iaux_2_data));
        CHECK(cudaFree(oaux_2_data));
        CHECK(cudaFree(aux_2_workload_gpu));
        free(aux_2_workload);
    }

    if(iaux_3_data){
        CHECK(cudaFree(iaux_3_data_gpu));
        CHECK(cudaFree(oaux_3_data_gpu));
        free(iaux_3_data);
        free(oaux_3_data);
    }

}