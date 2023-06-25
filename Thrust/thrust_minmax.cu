/////////////////////////////////////////////////////////////////////////////////////
/* THRUST MINMAX */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef THRUST_MINMAX
#define THRUST_MINMAX
#include "thrust_minmax.h"
#endif



struct compare_point
{
    __device__
    bool operator()(const Point& lhs, const Point& rhs) {
        return lhs.x < rhs.x;
    }
};


__global__ void assign_max_lines_thrust(Line* l_ptr, Point* min_ptr, Point* max_ptr){
    
    if(threadIdx.x == 0){
        l_ptr->p = *min_ptr;
        l_ptr->q = *max_ptr;
    }
}

void thrust_minmax(thrust::device_vector<Point>& points, thrust::device_vector<Line>& l){
    l.resize(1);
    thrust::device_vector<Point>::iterator iter_max = thrust::max_element(points.begin(), points.end(), compare_point());
    thrust::device_vector<Point>::iterator iter_min = thrust::min_element(points.begin(), points.end(), compare_point());

    Point* max_ptr = thrust::raw_pointer_cast(&(*iter_max)); // device pointer
    Point* min_ptr = thrust::raw_pointer_cast(&(*iter_min)); // device pointer

    Line* l_ptr = thrust::raw_pointer_cast(l.data());
    // cudaMemcpy(&(l_ptr->p), min_ptr, sizeof(Point), cudaMemcpyDeviceToDevice);
    // cudaMemcpy(&(l_ptr->q), max_ptr, sizeof(Point), cudaMemcpyDeviceToDevice);
    assign_max_lines_thrust<<<1, 1>>>(l_ptr, min_ptr, max_ptr);
}
