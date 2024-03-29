/////////////////////////////////////////////////////////////////////////////////////
/* DYNAMIC ARRAY */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef DYNAMIC_ARRAY
#define DYNAMIC_ARRAY
#include "dynamic_array.h"
#endif






void add_to_point_array(Point_array* points,  Point z){

    Point* new_array;

    if(points->curr_size > points->max_size/2){
        new_array = (Point*)realloc(points->array, (points->max_size << 1)*sizeof(Point));
        if(!new_array){
            fprintf(stderr, "Realloc failed");
            exit(1);
        }
        points->array = new_array;
        points->max_size <<= 1;  
    }


    int index = points->curr_size;
    memcpy(&points->array[index], &z, sizeof(Point));

    points->curr_size++;

}


void add_to_hull(Hull* hull, Line l){

    if(hull->curr_size > hull->max_size/2){
        hull->array = (Line*)realloc(hull->array, (hull->max_size << 1)*sizeof(Line));
        if(!hull->array){
            fprintf(stderr, "Realloc failed");
            exit(1);
        }
        hull->max_size <<= 1;  
    }


    int index = hull->curr_size;
    memcpy(&hull->array[index], &l, sizeof(Line));

    hull->curr_size++;

}


Hull* combine_hull(Hull* hull_1, Hull* hull_2){

    Hull* new_hull = (Hull*)malloc(sizeof(Hull));

    new_hull->curr_size = hull_1->curr_size+hull_2->curr_size;
    new_hull->max_size = new_hull->curr_size*2;

    new_hull->array = (Line*)malloc(new_hull->max_size*sizeof(Line));

    memcpy(new_hull->array, hull_1->array, hull_1->curr_size*sizeof(Line));
    memcpy(&new_hull->array[hull_1->curr_size], hull_2->array, hull_2->curr_size*sizeof(Line));

    free_hull(hull_1);
    free_hull(hull_2);

    return new_hull;

}


void free_point_array(Point_array* points){
    free(points->array);
    free(points);
}


void free_hull(Hull* hull){
    free(hull->array);
    free(hull);
}


void print_point_array(Point_array* points){
    fprintf(stdout, "Data: ");

    if(points->curr_size == 0){
        printf("\n");
        fflush(stdout);
        return;
    }

    for(int i = 0; i < points->curr_size-1; i++){
        fprintf(stdout, "(%lf,%lf), ", points->array[i].x, points->array[i].y);
    }
    fprintf(stdout, "(%lf,%lf)", points->array[points->curr_size-1].x, points->array[points->curr_size-1].y);
    fflush(stdout);
}