/////////////////////////////////////////////////////////////////////////////////////
/* MAX DISTANCE CUDA */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef MINMAX
#define MINMAX
#include "minmax.h"
#endif

#ifndef TEST
#define TEST
#include "../Test/test.h"
#endif

__global__ void minmax_kernel(Point* points, int size, Point* max, int op){
    __shared__ Point sdata[1024];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < size){
        sdata[tid] = points[gid];
    }else{
        sdata[tid] = points[0]; // dummy value
    }

    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            if (op == OP_MAX){
                //find Max
                if(sdata[tid].x < sdata[tid + s].x){
                    sdata[tid] = sdata[tid + s];
                }
            }else{
                //find Min
                if(sdata[tid].x > sdata[tid + s].x){
                    sdata[tid] = sdata[tid + s];
                }
            }
        }
        __syncthreads();
    }

    if(tid == 0){
        max[blockIdx.x] = sdata[0];
    }
}

//TODO: calculate both min and max, not only max.
void minmax_cuda(Point_array* points, Point* min, Point* max){
    int size = points->curr_size;
    int threadsPerBlock = 1024; //!!! always power of two and max 1024 because of static size of shared array in kernel !!!
    int numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;

    Point* d_points_in;
    CHECK(cudaMalloc((void**)&d_points_in, size * sizeof(Point)));
    CHECK(cudaMemcpy(d_points_in, points->array, size*sizeof(Point), cudaMemcpyHostToDevice));

    Point* d_points_out;
    CHECK(cudaMalloc((void**)&d_points_out, numBlocks * sizeof(Point)));

    while(size > threadsPerBlock){
        minmax_kernel<<<numBlocks, threadsPerBlock>>>(d_points_in, size, d_points_out, OP_MAX);
        CHECK(cudaMemcpy(d_points_in, d_points_out, numBlocks * sizeof(Point), cudaMemcpyDeviceToDevice));
        size = numBlocks;
        numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;
        CHECK(cudaFree(d_points_out));
        CHECK(cudaMalloc((void**)&d_points_out, numBlocks * sizeof(Point)));
    }

    Point* d_max;
    CHECK(cudaMalloc((void**)&d_max,sizeof(Point)));
    //deal with the rest
    minmax_kernel<<<1, threadsPerBlock>>>(d_points_in, size, d_max, OP_MAX);

    CHECK(cudaMemcpy(&max, d_max, sizeof(Point), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_points_in));
    CHECK(cudaFree(d_points_out));
}