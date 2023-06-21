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
    __shared__ Point sdata_min[BLOCKSIZE];
    __shared__ Point sdata_max[BLOCKSIZE];

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


__global__ void assign_max_lines_par(Line* l_ptr, Point* min_ptr, Point* max_ptr){

    if(threadIdx.x == 0){
        l_ptr->p = *min_ptr;
        l_ptr->q = *max_ptr;
    }
}

void minmax_cuda(Point_array_par* points, Line** l_pq){
    int size = points->size;
    int threadsPerBlock = BLOCKSIZE; //!!! always power of two and max 1024 because of static size of shared array in kernel !!!
    int numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;

    minmaxPoint points_in;
    CHECK(cudaMalloc((void**)&(points_in.max), size * sizeof(Point)));
    CHECK(cudaMalloc((void**)&(points_in.min), size * sizeof(Point)));
    CHECK(cudaMemcpy(points_in.max, points->array, size*sizeof(Point), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(points_in.min, points->array, size*sizeof(Point), cudaMemcpyDeviceToDevice));


    minmaxPoint points_out;
    CHECK(cudaMalloc((void**)&(points_out.max), numBlocks * sizeof(Point)));
    CHECK(cudaMalloc((void**)&(points_out.min), numBlocks * sizeof(Point)));

    while(size > threadsPerBlock){

        minmax_kernel<<<numBlocks, threadsPerBlock>>>(points_in, size, points_out);
        CHECK(cudaMemcpy(points_in.max, points_out.max, numBlocks * sizeof(Point), cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(points_in.min, points_out.min, numBlocks * sizeof(Point), cudaMemcpyDeviceToDevice));
        size = numBlocks;
        numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;
    }

    //deal with the rest
    minmax_kernel<<<1, threadsPerBlock>>>(points_in, size, points_out);

    // allocate GPU mem at address handed over as arguments
    CHECK(cudaMalloc(l_pq, sizeof(Line)));
    assign_max_lines_par<<<1, 1>>>(*l_pq, points_out.min, points_out.max);

    CHECK(cudaFree(points_out.min));
    CHECK(cudaFree(points_out.max));
    CHECK(cudaFree(points_in.min));
    CHECK(cudaFree(points_in.max));

}


///////////////////////////////////////////////////////////////////////////////
// Stream functions


void minmax_stream_cuda(Point_array_par* points, Line** l_pq, cudaStream_t* streams){
    int size = points->size;
    int threadsPerBlock = BLOCKSIZE; //!!! always power of two and max 1024 because of static size of shared array in kernel !!!
    int numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;

    minmaxPoint points_in;
    CHECK(cudaMallocAsync((void**)&(points_in.max), size * sizeof(Point), streams[0]));
    CHECK(cudaMallocAsync((void**)&(points_in.min), size * sizeof(Point), streams[1]));

    // Synchronice Kernels
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    CHECK(cudaMemcpyAsync(points_in.max, points->array, size*sizeof(Point), cudaMemcpyDeviceToDevice, streams[0]));
    CHECK(cudaMemcpyAsync(points_in.min, points->array, size*sizeof(Point), cudaMemcpyDeviceToDevice, streams[1]));


    minmaxPoint points_out;
    CHECK(cudaMallocAsync((void**)&(points_out.max), numBlocks * sizeof(Point), streams[0]));
    CHECK(cudaMallocAsync((void**)&(points_out.min), numBlocks * sizeof(Point), streams[1]));

    // Synchronice Kernels
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    while(size > threadsPerBlock){

        minmax_kernel<<<numBlocks, threadsPerBlock, 0, streams[0]>>>(points_in, size, points_out);

        // Synchronice Kernels
        cudaStreamSynchronize(streams[0]);

        CHECK(cudaMemcpyAsync(points_in.max, points_out.max, numBlocks * sizeof(Point), cudaMemcpyDeviceToDevice, streams[0]));
        CHECK(cudaMemcpyAsync(points_in.min, points_out.min, numBlocks * sizeof(Point), cudaMemcpyDeviceToDevice, streams[1]));
        size = numBlocks;
        numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;
        
        // Synchronice Kernels
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
    }

    //deal with the rest
    minmax_kernel<<<1, threadsPerBlock, 0, streams[0]>>>(points_in, size, points_out);

    // allocate GPU mem at address handed over as arguments
    CHECK(cudaMallocAsync(l_pq, sizeof(Line), streams[0]));
    assign_max_lines_par<<<1, 1, 0, streams[0]>>>(*l_pq, points_out.min, points_out.max);

    // Synchronice Kernels
    cudaStreamSynchronize(streams[0]);

    // free memory
    CHECK(cudaFreeAsync(points_in.min, streams[0]));
    CHECK(cudaFreeAsync(points_in.max, streams[1]));
    CHECK(cudaFreeAsync(points_out.min, streams[0]));
    CHECK(cudaFreeAsync(points_out.max, streams[1]));

    // Synchronice Kernels
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);


}

///////////////////////////////////////////////////////////////////////////////


//int main(int argc, char** argv){
//
//    int size = 100000000;
//
//    Point_array_par* points = init_point_array_par(size);
//
//    Point left = (Point){.x = -1, .y = 2};
//    Point middle = (Point){.x = 100, .y = 8};
//    Point right = (Point){.x = 200, .y = 3};
//
//    for(int i = 0; i < size; i++){
//        if(i == 9000000){
//            points->array[i] = left;
//        }else if(i == 1000000){
//            points->array[i] = right;
//        }else{
//            points->array[i] = middle;
//        }
//    }
//
//    Line* minmax;
//    minmax_cuda(points, &minmax);
//
//    Line minmax_h;
//    CHECK(cudaMemcpy(&minmax_h, minmax, sizeof(Line), cudaMemcpyDeviceToHost));
//
//    printf("minmax:\tp: (%f, %f)\tq: (%f, %f)\n", minmax_h.p.x, minmax_h.p.y, minmax_h.q.x, minmax_h.q.y);
//
//    return 0;
//}