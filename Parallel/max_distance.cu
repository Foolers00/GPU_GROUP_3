/////////////////////////////////////////////////////////////////////////////////////
/* MAX DISTANCE CUDA */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef MAX_DISTANCE
#define MAX_DISTANCE
#include "max_distance.h"
#endif

#ifndef TEST
#define TEST
#include "../Test/test.h"
#endif

struct PointDist{
    Point* p;
    double dist;
};

__device__ void distance_cuda(Line* l, Point* z, double* res){
    double a = l->p.y - l->q.y;
    double b = l->q.x - l->p.x;
    double c = l->p.x * l->q.y - l->q.x * l->p.y;
    // assert !(a == 0 || b == 0)
    *res = fabs(a * z->x  + b * z->y  + c)/sqrt(a * a + b * b);
}

__global__ void max_distance_kernel(Line* l, Point* points, int size, Point* max){

    __shared__ PointDist sdata[1024];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < size){
        Point* p = &points[gid];
        sdata[tid].p = p;
        distance_cuda(l, &points[gid], &(sdata[tid].dist));
    }else{
        sdata[tid] = (PointDist){.p = nullptr, .dist = 0};
    }

    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            if(sdata[tid].dist < sdata[tid + s].dist){
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if(tid == 0){
        max[blockIdx.x].x = sdata[0].p->x;
        max[blockIdx.x].y = sdata[0].p->y;
    }
}

Point max_distance_cuda(Line l, Point_array* points){
    int size = points->curr_size;
    int threadsPerBlock = 1024; //!!! always power of two and max 1024 because of static size of shared array in kernel !!!
    int numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;

    Line* d_l;
    CHECK(cudaMalloc((void**)&d_l, sizeof(Line)));
    CHECK(cudaMemcpy(d_l, &l, sizeof(Line), cudaMemcpyHostToDevice));

    Point* d_points_in;
    CHECK(cudaMalloc((void**)&d_points_in, size * sizeof(Point)));
    CHECK(cudaMemcpy(d_points_in, points->array, size*sizeof(Point), cudaMemcpyHostToDevice));

    Point* d_points_out;
    CHECK(cudaMalloc((void**)&d_points_out, numBlocks * sizeof(Point)));

    while(size > threadsPerBlock){
        max_distance_kernel<<<numBlocks, threadsPerBlock>>>(d_l, d_points_in, size, d_points_out);
        CHECK(cudaMemcpy(d_points_in, d_points_out, numBlocks * sizeof(Point), cudaMemcpyDeviceToDevice));
        size = numBlocks;
        numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;
        CHECK(cudaFree(d_points_out));
        CHECK(cudaMalloc((void**)&d_points_out, numBlocks * sizeof(Point)));
    }

    Point* d_max;
    CHECK(cudaMalloc((void**)&d_max,sizeof(Point)));
    //deal with the rest
    max_distance_kernel<<<1, threadsPerBlock>>>(d_l, d_points_in, size,d_max);

    Point max;
    CHECK(cudaMemcpy(&max, d_max, sizeof(Point), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_points_in));
    CHECK(cudaFree(d_l));
    CHECK(cudaFree(d_points_out));

    return max;
}


//int main(int argc, char** argv){
//    int size = 500000000; // 500 mio
//
//    Point_array* points = init_point_array(size);
//
//    Point near = (Point){.x = 1, .y = 2};
//    Point far = (Point){.x = 1, .y = 8};
//    Line l = (Line){.p = (Point){.x = 0, .y = 0}, .q = (Point){.x = 1000, .y = 1000}};
//
//    for(int i = 0; i < size; i++){
//        if(i == 9000000){
//            add_to_point_array(points, far);
//        }else{
//            add_to_point_array(points, near);
//        }
//    }
//
//    time_t tic = clock();
//    Point max_cuda = max_distance_cuda(l, points);
//    time_t toc = clock();
//    double sec_cuda = (double)(toc - tic)/CLOCKS_PER_SEC;
//    printf("Max dist cuda: (%f, %f) Time elapsed: %f\n", max_cuda.x, max_cuda.y, sec_cuda);
//
//    tic = clock();
//    Point max_seq = max_distance(l, points);
//    toc = clock();
//    double sec_seq = (double)(toc - tic)/CLOCKS_PER_SEC;
//    printf("Max dist seq: (%f, %f) Time elapsed: %f\n", max_seq.x, max_seq.y, sec_seq);
//
//    return 0;
//}