/////////////////////////////////////////////////////////////////////////////////////
/* THRUST GENERAL FUNCTIONS */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef THRUST_GENERAL_FUNCTIONS
#define THRUST_GENERAL_FUNCTIONS
#include "thrust_general_functions.h"
#endif



void thrust_quickhull(Point_array_par* points, thrust::host_vector<Line>& hull){

    // vars
    thrust::device_vector<Point> points_thrust;
    thrust::device_vector<Point> points_above;
    thrust::device_vector<Point> points_below;
    thrust::device_vector<Line> hull_up;
    thrust::device_vector<Line> hull_down;
    thrust::device_vector<Line> hull_result_thrust;
    
    // copy memory to gpu
    points_thrust.resize(points->size);
    thrust::copy(&points->array[0], &points->array[points->size], points_thrust.begin());


    // find points on hull

    // splits array into above and below
    thrust_split_point_array(points_thrust, points_above, points_below, Line l);

    // recursive call
    thrust_first_quickhull_split(points_above, hull_up, l, ABOVE);
    thrust_first_quickhull_split(points_above, hull_down, l, BELOW);

    // combine
    thrust_combine_hull(hull_up, hull_down, hull_result_thrust);

    // copy back results
    hull = hull_result_thrust;

    return;
}



void thrust_first_quickhull_split(thrust::device_vector<Point>& points, thrust::device_vector<Line>& hull,
                                  Line l, int side){
                                    
                                  }


void thrust_combine_hull(thrust::device_vector<Line>& hull_up, thrust::device_vector<Line>& hull_down, 
                         thrust::device_vector<Line>& hull_result){

    // resize 
    hull_result.resize(hull_up.size()+hull_down.size());

    // copy back results
    thrust::copy(hull_up.begin(), hull_up.end(), hull_result.begin());
    thrust::copy(hull_down.begin(), hull_down.end(), hull_result.begin()+hull_up.size());


}