/////////////////////////////////////////////////////////////////////////////////////
/* Data Types */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef DATA_TYPES
#define DATA_TYPES
#include "data_types.h"
#endif




Point_array* init_point_array(size_t max_size){
    
    Point_array* points;

    points = (Point_array*)malloc(sizeof(Point_array));
    if(!points){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    points->curr_size = 0;
    points->max_size = max_size;
    points->array = NULL;

    points->array = (Point*)malloc(points->max_size*sizeof(Point));
    if(!points->array){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    return points;

}


Hull* init_hull(size_t max_size){
    
    Hull* hull = NULL;
    
    hull = (Hull*)malloc(sizeof(Hull));
    if(!hull){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    hull->curr_size = 0;
    hull->max_size = max_size;
    hull->array = NULL;
    
    hull->array = (Line*)malloc(hull->max_size*sizeof(Line));
    if(!hull->array){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    return hull;

}



bool compare_lines(Line l_1, Line l_2){
    return (compare_points(l_1.p, l_2.p) && compare_points(l_1.q, l_2.q));
}



bool compare_points(Point p_1, Point p_2){
    return p_1.x == p_2.x && p_1.y == p_2.y;
}