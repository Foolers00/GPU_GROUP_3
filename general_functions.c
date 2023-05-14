/////////////////////////////////////////////////////////////////////////////////////
/* GENERAL FUNCTIONS */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef GENERAL_FUNCTIONS
#define GENERAL_FUNCTIONS
#include "general_functions.h"
#include <math.h>
#endif

Hull* quickhull(Point_array* points){

    Point p;
    Point q;
    Line l_pq;
    Hull* hull_up = NULL;
    Hull* hull_down = NULL;

    points_on_hull(points, &p, &q);
    l_pq = calc_line(p, q);

    hull_up = quickhull_split(points, l_pq, ABOVE);
    hull_down = quickhull_split(points, l_pq, BELOW);

    return combine_hull(hull_up, hull_down);

}

Hull* quickhull_split(Point_array* points, Line l, int side){
    Point_array* points_up = NULL;
    Point_array* points_down = NULL;
    Point max_up;
    Point max_down;
    Line l_p_up;
    Line l_q_up;
    Line l_p_down;
    Line l_q_down;
    Hull* hull_up = NULL;
    Hull* hull_down = NULL;

    // recursive abortion condition if 1
    if(points->curr_size == 1){
        init_hull(hull_up, 4);
        add_to_hull(hull_up, calc_line(l.p, points->array[0]));
        add_to_hull(hull_up, calc_line(l.q, points->array[0]));
        return hull_up;
    }

    init_point_array(points_up, points->max_size/2);
    init_point_array(points_down, points->max_size/2);

    for(int i = 0; i < points->curr_size; i++){
        int result = check_point_location(l, points->array[i]);
        if(result == ON || result != side){
            continue;
        }

        if(side){
            add_to_point_array(points_up, points->array[i]);
        }
        else{
            add_to_point_array(points_down, points->array[i]);
        }
    }

    // recursive abortion condition if 0
    if(points_up->curr_size){
        max_up = max_distance(l, points_up);
        l_p_up = calc_line(l.p, max_up);
        l_q_up = calc_line(l.q, max_up);
        hull_up = combine_hull(
                quickhull_split(points_up, l_p_up, ABOVE), 
                quickhull_split(points_up, l_q_up, ABOVE)
                );
    }
    else{
        init_hull(hull_up, 2);
        add_to_hull(hull_up, l);
    }


    if(points_down->curr_size){
        max_down = max_distance(l, points_down);
        l_p_down = calc_line(l.p, max_down);
        l_p_down = calc_line(l.q, max_down);
        hull_down = combine_hull(
                    quickhull_split(points_up, l_p_down, BELOW), 
                    quickhull_split(points_up, l_q_down, BELOW)
                    );
    }
    else{
        init_hull(hull_down, 2);
        add_to_hull(hull_down, l);
    }

    free_point_array(points_up);
    free_point_array(points_down);

    return combine_hull(hull_up, hull_down);

}


Line calc_line(Point p, Point q){

    double k;
    double d;
    Line l;

    k = (p.y-q.y)/(p.x-q.x);
    d = p.y - k*p.x;

    l.p = p;
    l.q = q;
    l.k = k;
    l.d = d;

    return l;

}

int check_point_location(Line l, Point z){

    if(l.k*z.x+l.d < z.y){
        return 1;
    }
    if(l.k*z.x+l.d == z.y){
        return 0;
    }
    return -1;

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
        exit(-1);
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

