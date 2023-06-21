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

    __shared__ PointDist sdata[BLOCKSIZE];

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


__global__ void assign_max_lines_par(Line* l_ptr, Line* l_p_max_ptr, Line* l_max_q_ptr, Point* max_ptr){

    if(threadIdx.x == 0){
        l_p_max_ptr->p = l_ptr->p;
        l_p_max_ptr->q = *max_ptr;

        l_max_q_ptr->p = *max_ptr;
        l_max_q_ptr->q = l_ptr->q;
    }
}


void max_distance_cuda(Line* l, Point_array_par* points, Line** l_p_max, Line** l_max_q){
    
    if(points->size == 0){
        return;
    }

    int size = points->size;
    int threadsPerBlock = BLOCKSIZE; //!!! always power of two and max 1024 because of static size of shared array in kernel !!!
    int numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;

    Point* d_points_in;
    CHECK(cudaMalloc((void**)&d_points_in, size * sizeof(Point)));
    CHECK(cudaMemcpy(d_points_in, points->array, size*sizeof(Point), cudaMemcpyDeviceToDevice));
    //d_points_in = points->array;

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

    assign_max_lines_par<<<1, 1>>>(l, *l_p_max, *l_max_q, d_max);

    CHECK(cudaFree(d_max));
    CHECK(cudaFree(d_points_in));
    CHECK(cudaFree(d_points_out));
}



///////////////////////////////////////////////////////////////////////////////
// Stream functions

void max_distance_stream_cuda(Line* l, Point_array_par* points, Line** l_p_max, Line** l_max_q, cudaStream_t* streams){
    
    if(points->size == 0){
        return;
    }

    int size = points->size;
    int threadsPerBlock = BLOCKSIZE; //!!! always power of two and max 1024 because of static size of shared array in kernel !!!
    int numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;

    Point* d_points_in;
    CHECK(cudaMallocAsync((void**)&d_points_in, size * sizeof(Point), streams[0]));
    CHECK(cudaMemcpyAsync(d_points_in, points->array, size*sizeof(Point), cudaMemcpyDeviceToDevice, streams[0]));
    //d_points_in = points->array;

    Point* d_points_out;
    CHECK(cudaMallocAsync((void**)&d_points_out, numBlocks * sizeof(Point), streams[0]));


    while(size > threadsPerBlock){
        max_distance_kernel<<<numBlocks, threadsPerBlock, 0, streams[0]>>>(l, d_points_in, size, d_points_out);
        CHECK(cudaMemcpyAsync(d_points_in, d_points_out, numBlocks * sizeof(Point), cudaMemcpyDeviceToDevice, streams[0]));
        size = numBlocks;
        numBlocks = (size + threadsPerBlock - 1)/threadsPerBlock;
        CHECK(cudaFreeAsync(d_points_out, streams[0]));
        CHECK(cudaMallocAsync((void**)&d_points_out, numBlocks * sizeof(Point), streams[0]));
    }

    Point* d_max;
    CHECK(cudaMallocAsync((void**)&d_max,sizeof(Point), streams[0]));
    //deal with the rest
    max_distance_kernel<<<1, threadsPerBlock, 0, streams[0]>>>(l, d_points_in, size,d_max);

    // allocate GPU mem at addresses handed over as arguments
    CHECK(cudaMallocAsync(l_p_max, sizeof(Line), streams[0]));
    CHECK(cudaMallocAsync(l_max_q, sizeof(Line), streams[0]));
    
    // assign lines
    assign_max_lines_par<<<1, 1, 0, streams[0]>>>(l, *l_p_max, *l_max_q, d_max);

    CHECK(cudaFreeAsync(d_max, streams[0]));
    CHECK(cudaFreeAsync(d_points_in, streams[0]));
    CHECK(cudaFreeAsync(d_points_out, streams[0]));
}



///////////////////////////////////////////////////////////////////////////////

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