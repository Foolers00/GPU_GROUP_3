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

void thrust_minmax(thrust::device_vector<Point>& points, thrust::device_vector<Line>& l){
    l.resize(1);
    thrust::device_vector<Point>::iterator iter_max = thrust::max_element(points.begin(), points.end(), compare_point());
    thrust::device_vector<Point>::iterator iter_min = thrust::min_element(points.begin(), points.end(), compare_point());

    Point* max_ptr = thrust::raw_pointer_cast(&(*iter_max)); // device pointer
    Point* min_ptr = thrust::raw_pointer_cast(&(*iter_min)); // device pointer

    Line* l_ptr = thrust::raw_pointer_cast(l.data());
    cudaMemcpy(&(l_ptr->p), min_ptr, sizeof(Point), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&(l_ptr->q), max_ptr, sizeof(Point), cudaMemcpyDeviceToDevice);
}

//int main(int argc, char** argv){
//    int size = 100000000;
//
//    Point_array_par* points = init_point_array_par(size);
//
//    Point left = (Point){.x = -1, .y = 2};
//    Point middle = (Point){.x = 100, .y = 8};
//    Point right = (Point){.x = 200, .y = 3};
//
//    for(int i = 0; i < size; i++) {
//        if (i == size/3) {
//            points->array[i] = left;
//        } else if (i == (size/2+size/3)) {
//            points->array[i] = right;
//        } else {
//            points->array[i] = middle;
//        }
//    }
//
//    thrust::device_vector<Point> d_points(size);
//    d_points.resize(size);
//    thrust::copy(&points->array[0], &points->array[size], d_points.begin());
//
//    thrust::device_vector<Line> l_d(1);
//
//    thrust_minmax(d_points, l_d);
//
//    thrust::host_vector<Line> l_h = l_d;
//
//    printf("line l: (%f, %f) - (%f, %f)\n", static_cast<Line>(l_h[0]).p.x, static_cast<Line>(l_h[0]).p.y, static_cast<Line>(l_h[0]).q.x, static_cast<Line>(l_h[0]).q.y);
//
//}