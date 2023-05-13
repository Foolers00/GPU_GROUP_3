/////////////////////////////////////////////////////////////////////////////////////
/* GENERAL FUNCTIONS */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef GENERAL_FUNCTIONS
#define GENERAL_FUNCTIONS
#include "general_functions.h"
#include <math.h>
#endif




bool check_above(Line l, Point z){
    if(l.k*z.x+l.d < z.y){
        return true;
    }
    return false;
}


void generate_random_points(Point_array* points, double l_bound, double u_bound){

    time_t t;
    double difference = u_bound - l_bound;
    double offset_x = 0;
    double offset_y = 0;

    srand((unsigned) time(&t));

    for(int i = 0; i < points->curr_size; i++){
        offset_x = rand() % (int)difference;
        offset_y = rand() % (int)difference;
        points->array[i].x = l_bound + offset_x;
        points->array[i].y = l_bound + offset_y;
    }

    points->index = points->curr_size;

}


void points_on_hull(Point_array* points, Point* p, Point* q){
    Point max;
    Point min;

    max.x = 0;
    min.x = DBL_MAX;

    for(int i = 0; i < points->curr_size; i++){
        if(points->array[i].x > max.x){
            max.x = points->array[i].x;
            max.y = points->array[i].y;
        }

        if(points->array[i].x < min.x){
            min.x = points->array[i].x;
            min.y = points->array[i].y;
        }
    }

    p->x = min.x;
    p->y = min.y;

    q->x = max.x;
    q->y = max.y;
}

// calculates the minimal distance from a Point z to the Line l
double distance(Line l, Point z){
    double a = l.k;
    double b = -1.0;
    double c = l.d;
    if (a == 0 && b == 0) {
        fprintf(stderr, "distance cannot be calculated for illegal line equation.");
        exit -1;
    }else{
        return fabs(a * z.x  + b * z.y  + c)/sqrt(a * a + b *b );
    }
}

// returns the Point with maximal distance to the Line l
Point max_distance(Line l, Point_array* points){
    double max = -1.0;
    Point p;
    for (int i = 0; i < points->curr_size; i++){
        double dist = distance(l, points->array[i]);
        if(dist > max){
            max = dist;
            p = points->array[i];
        }
    }
    return p;
}
