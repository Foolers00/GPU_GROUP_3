/////////////////////////////////////////////////////////////////////////////////////
/* THRUST SPLIT */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef THRUST_SPLIT
#define THRUST_SPLIT
#include "thrust_split.h"
#endif


struct check_point_location_functor
{
    thrust::device_vector<Line>& l;
    int side;
   
    check_point_location_functor(thrust::device_vector<Line>&_l, int _side) : l(_l), side(_side) { }

    __device__
        bool operator()(const Point& point){
            
            int result;
            check_point_location_thrust(l[0], point, &result);
            return (result == side);
                
        }

};



void thrust_split_point_array(thrust::device_vector<Point>& points, thrust::device_vector<Point>& points_above, 
                                thrust::device_vector<Point>&  points_below, thrust::device_vector<Line>& l){

        // copy to gpu
        //thrust::device_vector<Point> points_gpu = points;


        // resize space
        points_above.resize(points.size());
        points_below.resize(points.size());


        thrust::make_zip_iterator(thrust::make_tuple(points.begin(), l.begin()), thrust::make_tuple(points.begin(), l.begin()));

        // move values
        auto points_above_end = thrust::copy_if(thrust::device, points.begin(), points.end(), points_above.begin(), check_point_location_functor(l, ABOVE));
        auto points_below_end = thrust::copy_if(thrust::device, points.begin(), points.end(), points_below.begin(), check_point_location_functor(l, BELOW));

        // set size
        points_above.resize(points_above_end-points_above.begin());
        points_below.resize(points_below_end-points_below.begin());
        
}


void thrust_split_point_array_side(thrust::device_vector<Point>& points, thrust::device_vector<Point>& points_side, 
                                    thrust::device_vector<Line>& l, int side){

        // resize space
        points_side.resize(points.size());

        // move values
        auto points_side_end = thrust::copy_if(thrust::device, points.begin(), points.end(), points_side.begin(), check_point_location_functor(l, side));

        // set size
        points_side.resize(points_side_end-points_side.begin());

    }



__device__ void check_point_location_thrust(Line l, Point z, int* result){

    Vector v1;
    Vector v2; 
    
    double cross_result;

    init_vector_thrust(l.p, l.q, &v1);
    init_vector_thrust(l.p, z, &v2);
    
    cross_product_thrust(v1, v2, &cross_result);

    if(cross_result>0){
        *result = ABOVE;
        return;
    }
    if(cross_result == 0){
        *result = ON;
        return;
    }
    *result = BELOW;
    return;

}

__device__ Vector init_vector_thrust(Point p, Point q, Vector* v){

    v->_x = q.x-p.x;
    v->_y = q.y-p.y;

    return;
}


__device__ void cross_product_thrust(Vector v1, Vector v2, double* result){
    
    *result = (v1._x*v2._y)-(v1._y*v2._x);
    return;

}


