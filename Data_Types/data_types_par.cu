/////////////////////////////////////////////////////////////////////////////////////
/* Data Types Parallel */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef DATA_TYPES_PAR
#define DATA_TYPES_PAR
#include "data_types_par.h"
#endif




Point_array_par* init_point_array_par(size_t size){
    
    Point_array_par* points;

    points = (Point_array_par*)malloc(sizeof(Point_array_par));
    if(!points){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    points->size = size;
    points->array = NULL;

    points->array = (Point*)malloc(points->size*sizeof(Point));
    if(!points->array){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    return points;

}



Hull_par* init_hull_par(size_t size){
    
    Hull_par* hull = NULL;
    
    hull = (Hull_par*)malloc(sizeof(Hull_par));
    if(!hull){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    hull->size = size;
    hull->array = NULL;
    
    hull->array = (Line*)malloc(hull->size*sizeof(Line));
    if(!hull->array){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }

    return hull;

}