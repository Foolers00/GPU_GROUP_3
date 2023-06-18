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
    thrust::device_vector<Line> l;
    
    // copy memory to gpu
    points_thrust.resize(points->size);
    thrust::copy(&points->array[0], &points->array[points->size], points_thrust.begin());


    // find points on hull
    thrust_minmax(points_thrust, l);
    thrust::host_vector<Line> l_h = l;
    printf("(%f, %f) - (%f, %f) \n", static_cast<Line>(l_h[0]).p.x, static_cast<Line>(l_h[0]).p.y, static_cast<Line>(l_h[0]).q.x, static_cast<Line>(l_h[0]).q.y);

    // splits array into above and below
    thrust_split_point_array(points_thrust, points_above, points_below, l);

    // recursive call
    thrust_first_quickhull_split(points_above, hull_up, l, ABOVE);
    thrust_first_quickhull_split(points_above, hull_down, l, BELOW);

    // combine
    thrust_combine_hull(hull_up, hull_down, hull_result_thrust);

    // copy back results
    hull = hull_result_thrust;

    return;
}



void thrust_first_quickhull_split(thrust::device_vector<Point>& points, thrust::device_vector<Line>& hull_side,
                                  thrust::device_vector<Line>& l, int side){
    
    // vars
    thrust::device_vector<Point>& points_side = points;
    thrust::device_vector<Line> l_max; // l_max[0] = l_p_max,  l_max[1] = l_max_q
    thrust::device_vector<Line> hull_side_1;
    thrust::device_vector<Line> hull_side_2;



    // find point with max distance
    thrust_max_distance(l, points_side, l_max);
    ////////////////////////////////



    if(points_side.size() == 0){
        hull_side = l;
    }
    else if(points_side.size() == 1){
        hull_side = l_max;
    }
    else{
        thrust::device_vector<Line> l_p_max(l_max.begin(), l_max.begin());
        thrust::device_vector<Line> l_max_q(l_max.begin()+1, l_max.begin()+1);

        thrust_quickhull_split(points_side, hull_side_1, l_p_max, side);
        thrust_quickhull_split(points_side, hull_side_2, l_max_q, side);
        thrust_combine_hull(hull_side_1, hull_side_2, hull_side);
    }
}

                            
void thrust_quickhull_split(thrust::device_vector<Point>& points, thrust::device_vector<Line>& hull_side,
                                  thrust::device_vector<Line>& l, int side){

    // vars
    thrust::device_vector<Point> points_side;
    thrust::device_vector<Line> l_max; // l_max[0] = l_p_max,  l_max[1] = l_max_q
    thrust::device_vector<Line> hull_side_1;
    thrust::device_vector<Line> hull_side_2;

    // set memory
    points_side.resize(points.size());


    // split array
    thrust_split_point_array_side(points, points_side, l, side);


    // find point with max distance
    ////////////////////////////////

    
    

    if(points_side.size() == 0){
        hull_side = l;
    }
    else if(points_side.size() == 1){
        hull_side = l_max;
    }
    else{
        thrust::device_vector<Line> l_p_max(l_max.begin(), l_max.begin());
        thrust::device_vector<Line> l_max_q(l_max.begin()+1, l_max.begin()+1);

        thrust_quickhull_split(points_side, hull_side_1, l_p_max, side);
        thrust_quickhull_split(points_side, hull_side_2, l_max_q, side);
        thrust_combine_hull(hull_side_1, hull_side_2, hull_side);
    }                    
}





void thrust_combine_hull(thrust::device_vector<Line>& hull_up, thrust::device_vector<Line>& hull_down, 
                         thrust::device_vector<Line>& hull_result){

    // resize 
    hull_result.resize(hull_up.size()+hull_down.size());

    // copy back results
    thrust::copy(hull_up.begin(), hull_up.end(), hull_result.begin());
    thrust::copy(hull_down.begin(), hull_down.end(), hull_result.begin()+hull_up.size());


}