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

void max_distance_cuda(Line* l, Point_array_par* points, Line** l_p_max, Line** l_max_q){
    int size = points->size;
    int threadsPerBlock = 1024; //!!! always power of two and max 1024 because of static size of shared array in kernel !!!
    int numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;

//    Line* d_l;
//    CHECK(cudaMalloc((void**)&d_l, sizeof(Line)));
//    CHECK(cudaMemcpy(d_l, &l, sizeof(Line), cudaMemcpyHostToDevice));

    Point* d_points_in;
    CHECK(cudaMalloc((void**)&d_points_in, size * sizeof(Point)));
    CHECK(cudaMemcpy(d_points_in, points->array, size*sizeof(Point), cudaMemcpyHostToDevice));

    Point* d_points_out;
    CHECK(cudaMalloc((void**)&d_points_out, numBlocks * sizeof(Point)));

    while(size > threadsPerBlock){
        max_distance_kernel<<<numBlocks, threadsPerBlock>>>(l, d_points_in, size, d_points_out);
        CHECK(cudaMemcpy(d_points_in, d_points_out, numBlocks * sizeof(Point), cudaMemcpyDeviceToDevice));
        size = numBlocks;
        numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;
        CHECK(cudaFree(d_points_out));
        CHECK(cudaMalloc((void**)&d_points_out, numBlocks * sizeof(Point)));
    }

    Point* d_max;
    CHECK(cudaMalloc((void**)&d_max,sizeof(Point)));
    //deal with the rest
    max_distance_kernel<<<1, threadsPerBlock>>>(l, d_points_in, size,d_max);

    // allocate GPU mem at addresses handed over as arguments
    CHECK(cudaMalloc(l_p_max, sizeof(Line)));
    CHECK(cudaMalloc(l_max_q, sizeof(Line)));
    CHECK(cudaMemcpy(&(*l_p_max)->p, &l->p, sizeof(Point), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(&(*l_p_max)->q, d_max, sizeof(Point), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(&(*l_max_q)->p, d_max, sizeof(Point), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(&(*l_max_q)->q, &l->q, sizeof(Point), cudaMemcpyDeviceToDevice));

    CHECK(cudaFree(d_points_in));
    CHECK(cudaFree(d_points_out));
}


//int main(int argc, char** argv){
//    int size = 100000000;
//
//    Point_array_par* points = init_point_array_par(size);
//
//    Point near = (Point){.x = 1, .y = 2};
//    Point far = (Point){.x = 1, .y = 8};
//    Line l = (Line){.p = (Point){.x = 0, .y = 0}, .q = (Point){.x = 1000, .y = 1000}};
//
//    for(int i = 0; i < size; i++) {
//        points->array[i] = near;
//        if (i == 1234) points->array[i] = far;
//    }
//
//    Line* d_l;
//    Line* l_p_max, *l_max_q;
//
//    CHECK(cudaMalloc((void**)&d_l, sizeof(Line)));
//    CHECK(cudaMemcpy(d_l, &l, sizeof(Line), cudaMemcpyHostToDevice));
//    max_distance_cuda(d_l, points, &l_p_max, &l_max_q);
//
//    Line l_p_max_host, l_max_q_host;
//
//    CHECK(cudaMemcpy(&l_p_max_host, l_p_max, sizeof(Line), cudaMemcpyDeviceToHost));
//    CHECK(cudaMemcpy(&l_max_q_host, l_max_q, sizeof(Line), cudaMemcpyDeviceToHost));
//    printf("l_p_max:\tp: (%f, %f)\tq: (%f, %f)\n", l_p_max_host.p.x, l_p_max_host.p.y, l_p_max_host.q.x, l_p_max_host.q.y);
//    printf("l_max_q:\tp: (%f, %f)\tq: (%f, %f)\n", l_max_q_host.p.x, l_max_q_host.p.y, l_max_q_host.q.x, l_max_q_host.q.y);
//
//
//    return 0;
//}