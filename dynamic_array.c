/////////////////////////////////////////////////////////////////////////////////////
/* DYNAMIC ARRAY */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef DYNAMIC_ARRAY
#define DYNAMIC_ARRAY
#include "dynamic_array.h"
#endif



void init_point_array(Point_array* points, size_t max_size){
    
    points->curr_size = 0;
    points->max_size = max_size;
    points->index = 0;
    points->array = NULL;

    points->array = (Point*)malloc(points->max_size*sizeof(Point));
    if(!points->array){
        fprintf(stderr, "Malloc failed");
    }

}


void init_hull(Hull* hull, size_t max_size){
    
    hull->curr_size = 0;
    hull->max_size = max_size;
    hull->index = 0;
    hull->array = NULL;
    
    hull->array = (Line*)malloc(hull->max_size*sizeof(Line));
    if(!hull->array){
        fprintf(stderr, "Malloc failed");
    }

}

void add_to_point_array(Point_array* points,  Point z){

    if(points->curr_size > points->max_size/2){
        points->array = (Point*)realloc(points->array, points->max_size << 1);
        if(!points->array){
            fprintf(stderr, "Realloc failed");
            exit(1);
        }
        points->max_size <<= 1;  
    }


    int index = points->index;
    memcpy(&points->array[index], &z, sizeof(Point));

    points->index++;
    points->curr_size++;

}


void add_to_hull(Hull* hull, Line l){

    if(hull->curr_size > hull->max_size/2){
        hull->array = (Line*)realloc(hull->array, hull->max_size << 1);
        if(!hull->array){
            fprintf(stderr, "Realloc failed");
            exit(1);
        }
        hull->max_size <<= 1;  
    }


    int index = hull->index;
    memcpy(&hull->array[index], &l, sizeof(Line));

    hull->index++;
    hull->curr_size++;

}


Hull* combine_hull(Hull* hull_1, Hull* hull_2){

    Hull* new_hull = (Hull*)malloc(sizeof(Hull));

    new_hull->curr_size = hull_1->curr_size+hull_2->curr_size;

    if(hull_1->max_size >= hull_2->max_size){
        new_hull->max_size = hull_1->max_size<1;
    }
    else{
        new_hull->max_size = hull_2->max_size<1;
    }

    new_hull->array = (Line*)malloc(new_hull->max_size*sizeof(Line));

    memcpy(new_hull->array, hull_1->array, hull_1->curr_size);
    memcpy(&new_hull->array[hull_1->curr_size], hull_2->array, hull_2->curr_size);

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
    for(int i = 0; i < points->curr_size-1; i++){
        fprintf(stdout, "(%lf,%lf), ", points->array[i].x, points->array[i].y);
    }
    fprintf(stdout, "(%lf,%lf), ", points->array[points->curr_size-1].x, points->array[points->curr_size-1].y);
}