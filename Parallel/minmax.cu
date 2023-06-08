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


__global__ void minmax_kernel(minmaxPoint points, int size, minmaxPoint result){

    // 1024 * 16 * 2 = 32,8 KB
    __shared__ Point sdata_min[1024];
    __shared__ Point sdata_max[1024];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < size){
        sdata_min[tid] = points.min[gid];
        sdata_max[tid] = points.max[gid];
    }else{
        sdata_min[tid] = points.min[0]; // dummy value
        sdata_max[tid] = points.max[0]; // dummy value
    }

    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            //find Max
            if(sdata_max[tid].x < sdata_max[tid + s].x){
                sdata_max[tid] = sdata_max[tid + s];
            }
            //find Min
            if(sdata_min[tid].x > sdata_min[tid + s].x) {
                sdata_min[tid] = sdata_min[tid + s];
            }
        }
        __syncthreads();
    }

    if(tid == 0){
        result.min[blockIdx.x] = sdata_min[0];
        result.max[blockIdx.x] = sdata_max[0];
    }
}

//TODO: calculate both min and max, not only max.
void minmax_cuda(Point_array* points, Point* min, Point* max){
    int size = points->curr_size;
    int threadsPerBlock = 1024; //!!! always power of two and max 1024 because of static size of shared array in kernel !!!
    int numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;

    minmaxPoint points_in;
    CHECK(cudaMalloc((void**)&(points_in.max), size * sizeof(Point)));
    CHECK(cudaMalloc((void**)&(points_in.min), size * sizeof(Point)));
    CHECK(cudaMemcpy(points_in.max, points->array, size*sizeof(Point), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(points_in.min, points->array, size*sizeof(Point), cudaMemcpyHostToDevice));


    minmaxPoint points_out;
    CHECK(cudaMalloc((void**)&(points_out.max), numBlocks * sizeof(Point)));
    CHECK(cudaMalloc((void**)&(points_out.min), numBlocks * sizeof(Point)));

    while(size > threadsPerBlock){

        minmax_kernel<<<numBlocks, threadsPerBlock>>>(points_in, size, points_out);
        CHECK(cudaMemcpy(points_in.max, points_out.max, numBlocks * sizeof(Point), cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(points_in.min, points_out.min, numBlocks * sizeof(Point), cudaMemcpyDeviceToDevice));
        size = numBlocks;
        numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;
//        CHECK(cudaFree(d_points_out->min));
//        CHECK(cudaFree(d_points_out->max));
//        CHECK(cudaFree(d_points_out));
//        CHECK(cudaMalloc((void**)&d_points_out_min, numBlocks * sizeof(Point)));
//        CHECK(cudaMalloc((void**)&d_points_out_max, numBlocks * sizeof(Point)));
    }

    //deal with the rest
    minmax_kernel<<<1, threadsPerBlock>>>(points_in, size, points_out);

    CHECK(cudaMemcpy(max, points_out.max, sizeof(Point), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(min, points_out.min, sizeof(Point), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(points_out.min));
    CHECK(cudaFree(points_out.max));
    CHECK(cudaFree(points_in.min));
    CHECK(cudaFree(points_in.max));

}
//int main(int argc, char **argv){
//    int size = 450000000; // 450 mio
//
//    Point_array* points = init_point_array(size);
//
//    Point left = (Point){.x = -1, .y = 2};
//    Point middle = (Point){.x = 100, .y = 8};
//    Point right = (Point){.x = 200, .y = 3};
//
//    for(int i = 0; i < size; i++){
//        if(i == 90000000){ // 90 Mio
//            add_to_point_array(points, left);
//        }else if(i == 1000000){ // 1 Mio
//            add_to_point_array(points, right);
//        }else{
//            add_to_point_array(points, middle);
//        }
//    }
//
//    time_t tic = clock();
//    Point max_cuda, min_cuda;
//    minmax_cuda(points, &min_cuda, &max_cuda);
//    time_t toc = clock();
//    double sec_cuda = (double)(toc - tic)/CLOCKS_PER_SEC;
//    printf("Max cuda: (%f, %f), Min cuda: (%f, %f), Time elapsed: %f\n", max_cuda.x, max_cuda.y, min_cuda.x, min_cuda.y, sec_cuda);
//
//    tic = clock();
//    Point max_seq, min_seq;
//    points_on_hull(points, &min_seq, &max_seq);
//    toc = clock();
//    double sec_seq = (double)(toc - tic)/CLOCKS_PER_SEC;
//    printf("Max seq: (%f, %f), Min seq: (%f, %f), Time elapsed: %f\n", max_seq.x, max_seq.y, min_seq.x, min_seq.y, sec_seq);
//
//    return 0;
//}
