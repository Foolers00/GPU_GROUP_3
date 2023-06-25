/////////////////////////////////////////////////////////////////////////////////////
/* THRUST MAX DISTANCE */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef THRUST_MAX_DISTANCE
#define THRUST_MAX_DISTANCE
#include "thrust_max_distance.h"
#endif


struct distance_functor
{
    Line* l;

    distance_functor(Line* _l) : l(_l) {}

    __device__
    bool operator()(const Point& lhs, const Point& rhs) {
        double res_lhs, res_rhs;
        dist(l, lhs, res_lhs);
        dist(l, rhs, res_rhs);
        return res_lhs < res_rhs;
    }
};

__device__ void dist(Line* l, const Point& z, double& res){
    double a = l->p.y - l->q.y;
    double b = l->q.x - l->p.x;
    double c = l->p.x * l->q.y - l->q.x * l->p.y;
    // assert !(a == 0 || b == 0)
    res = fabs(a * z.x  + b * z.y  + c)/sqrt(a * a + b * b);
}


__global__ void assign_max_lines_thrust(Line* l_ptr, Line* l_p_max_ptr, Line* l_max_q_ptr, Point* max_ptr){
    
    if(threadIdx.x == 0){
        l_p_max_ptr->p = l_ptr->p;
        l_p_max_ptr->q = *max_ptr;

        l_max_q_ptr->p = *max_ptr;
        l_max_q_ptr->q = l_ptr->q;
    }
}


void thrust_max_distance(thrust::device_vector<Line>& l, thrust::device_vector<Point>& points, thrust::device_vector<Line>& l_max){
    l_max.resize(2);
    Line* l_ptr = thrust::raw_pointer_cast(l.data());
    thrust::device_vector<Point>::iterator iter = thrust::max_element(points.begin(), points.end(), distance_functor(l_ptr));
    Point* max_ptr = thrust::raw_pointer_cast(&(*iter)); //device pointer

    Line* l_p_max_ptr = thrust::raw_pointer_cast(&(*l_max.begin())); //device pointer
    Line* l_max_q_ptr = thrust::raw_pointer_cast(&(*(l_max.begin() + 1))); //device pointer

    assign_max_lines_thrust<<<1, 1>>>(l_ptr, l_p_max_ptr, l_max_q_ptr, max_ptr);

}
