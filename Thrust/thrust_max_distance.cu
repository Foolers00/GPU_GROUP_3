/////////////////////////////////////////////////////////////////////////////////////
/* THRUST MAX DISTANCE */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef THRUST_SPLIT
#define THRUST_SPLIT
#include "thrust_max_distance.h"
#endif

struct distance_functor
{
    Line* l;

    distance_functor(Line* _l) : l(_l) {}

    __device__
    float operator()(const Point& lhs, const Point& rhs) {
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


void thrust_max_distance(thrust::device_vector<Line>& l, thrust::device_vector<Point>& points, thrust::device_vector<Line>& l_max){

    Line* l_ptr = thrust::raw_pointer_cast(l.data());
    thrust::device_vector<Point>::iterator iter = thrust::max_element(points.begin(), points.end(), distance_functor(l_ptr));
    Point max = *iter;
    Point* max_ptr = thrust::raw_pointer_cast(&(*iter)); //device pointer

    Line* l_p_max_ptr = thrust::raw_pointer_cast(&(*l_max.begin())); //device pointer
    Line* l_max_q_ptr = thrust::raw_pointer_cast(&(*(l_max.begin() + 1))); //device pointer
    cudaMemcpy(&(l_p_max_ptr->p), &(l_ptr->p), sizeof(Point), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&(l_p_max_ptr->q), max_ptr, sizeof(Point), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&(l_max_q_ptr->p), max_ptr, sizeof(Point), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&(l_max_q_ptr->q), &(l_ptr->q), sizeof(Point), cudaMemcpyDeviceToDevice);

}

//int main(int argc, char** argv){
//    int size = 100000000;
//
//    Point_array_par* points = init_point_array_par(size);
//
//    Point near = (Point){.x = 1, .y = 2};
//    Point far = (Point){.x = 1, .y = 22};
//    Line l = (Line){.p = (Point){.x = 1, .y = 1}, .q = (Point){.x = 1000, .y = 1000}};
//
//    for(int i = 0; i < size; i++) {
//        points->array[i] = near;
//        if (i == 1230000) points->array[i] = far;
//    }
//
//    thrust::host_vector<Line> l_h(1);
//    l_h[0] = l;
//    thrust::device_vector<Line> l_d = l_h;
//    thrust::device_vector<Line> l_max(2);
//    thrust::device_vector<Point> d_points(size);
//    d_points.resize(size);
//    thrust::copy(&points->array[0], &points->array[size], d_points.begin());
//    thrust_max_distance(l_d, d_points, l_max);
//
//    thrust::host_vector<Line> l_max_h = l_max;
//
//
//    printf("line p_max: (%f, %f) - (%f, %f)\n", static_cast<Line>(l_max_h[0]).p.x, static_cast<Line>(l_max_h[0]).p.y, static_cast<Line>(l_max_h[0]).q.x, static_cast<Line>(l_max_h[0]).q.y);
//    printf("line max_q: (%f, %f) - (%f, %f)\n", static_cast<Line>(l_max_h[1]).p.x, static_cast<Line>(l_max_h[1]).p.y, static_cast<Line>(l_max_h[1]).q.x, static_cast<Line>(l_max_h[1]).q.y);
//
//
//}