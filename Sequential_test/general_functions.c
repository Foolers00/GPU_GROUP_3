/////////////////////////////////////////////////////////////////////////////////////
/* GENERAL FUNCTIONS */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef GENERAL_FUNCTIONS
#define GENERAL_FUNCTIONS
#include "general_functions.h"
#endif

Hull* quickhull(Point_array* points){

    Point p;
    Point q;
    Line l_pq;
    Hull* hull_up = NULL;
    Hull* hull_down = NULL;

    points_on_hull(points, &p, &q);
    l_pq = (Line) { .p = p, .q = q };

    //printf("Line: (%f, %f)-(%f, %f)\n", l_pq.p.x, l_pq.p.y, l_pq.q.x, l_pq.q.y);

    hull_up = quickhull_split(points, l_pq, ABOVE);
    hull_down = quickhull_split(points, l_pq, BELOW);

    return combine_hull(hull_up, hull_down);

}


Hull* new_quickhull(Point_array* points){

    Point p;
    Point q;
    Line l_pq;
    Hull* hull_up = NULL;
    Hull* hull_down = NULL;
    Point_array* points_above;
    Point_array* points_below;

    points_on_hull(points, &p, &q);
    l_pq = (Line) { .p = p, .q = q };

    points_above = init_point_array(points->max_size/2);
    points_below = init_point_array(points->max_size/2);

    for(int i = 0; i < points->curr_size; i++){
        int result = check_point_location(l_pq, points->array[i]);
        if(result == ON){
            continue;
        }
        else if(result == ABOVE){
            add_to_point_array(points_above, points->array[i]);
        }
        else{
            add_to_point_array(points_below, points->array[i]);
        }
    }

    hull_up = first_quickhull_split(points_above, l_pq, ABOVE);
    hull_down = first_quickhull_split(points_below, l_pq, BELOW);

    return combine_hull(hull_up, hull_down);

}


Hull* first_quickhull_split(Point_array* points, Line l, int side){
    
    Point_array* points_side = NULL;
    Point max_point;
    Line l_p_max;
    Line l_max_q;
    Hull* hull_side = NULL;


    points_side = points;

    

    max_point = max_distance(l, points_side);
    l_p_max = (Line) { .p = l.p, .q = max_point };
    l_max_q = (Line) { .p = max_point, .q = l.q };

    if(points_side->curr_size == 0) {
        hull_side = init_hull(2);
        add_to_hull(hull_side, l);
    }else if(points_side->curr_size == 1){
        hull_side = init_hull(4);
        add_to_hull(hull_side, l_p_max);
        add_to_hull(hull_side, l_max_q);
    }else {
        //points_side->curr_size > 1
        hull_side = combine_hull(
                quickhull_split(points_side, l_p_max, side),
                quickhull_split(points_side, l_max_q, side)
        );
    }

    return hull_side;

}

Hull* quickhull_split(Point_array* points, Line l, int side){
    
    Point_array* points_side = NULL;
    Point max_point;
    Line l_p_max;
    Line l_max_q;
    Hull* hull_side = NULL;

    points_side = init_point_array(points->max_size/2);

    for(int i = 0; i < points->curr_size; i++){
        int result = check_point_location(l, points->array[i]);
        if(result == ON || result != side){
            continue;
        }

        if(side){
            add_to_point_array(points_side, points->array[i]);
        }
    }

    //print_point_array(points_side);

    max_point = max_distance(l, points_side);
    l_p_max = (Line) { .p = l.p, .q = max_point };
    l_max_q = (Line) { .p = max_point, .q = l.q };

    // printf("l_p_max: (%f, %f)-(%f, %f)\n", l_p_max.p.x, l_p_max.p.y, l_p_max.q.x, l_p_max.q.y);
    // printf("l_max_q: (%f, %f)-(%f, %f)\n", l_max_q.p.x, l_max_q.p.y, l_max_q.q.x, l_max_q.q.y);

    if(points_side->curr_size == 0) {
        hull_side = init_hull(2);
        add_to_hull(hull_side, l);
    }else if(points_side->curr_size == 1){
        hull_side = init_hull(4);
        add_to_hull(hull_side, l_p_max);
        add_to_hull(hull_side, l_max_q);
    }else {
        //points_side->curr_size > 1
        hull_side = combine_hull(
                quickhull_split(points_side, l_p_max, side),
                quickhull_split(points_side, l_max_q, side)
        );
    }
    free_point_array(points_side);
    return hull_side;

}



Line init_line(Point p, Point q){

    Line l;

    l.p = p;
    l.q = q;

    return l;

}


int check_point_location(Line l, Point z){

    double result = cross_product(init_vector(l.p, l.q), init_vector(l.p, z));

    if(fabs(result) < ZERO_PRECISION){
        return ON;
    } 
    if(result>0){
        return ABOVE;
    }
    return BELOW;

}


Point_array* generate_random_points(int num_of_points, double l_bound, double u_bound){

    time_t t;
    double difference = u_bound - l_bound;
    double offset_x = 0;
    double offset_y = 0;
    srand((unsigned) time(&t));

    Point_array* points = init_point_array(num_of_points * 2);
    for(size_t i = 0; i < num_of_points; i++){
        offset_x = rand() % (int)difference;
        offset_y = rand() % (int)difference;
        add_to_point_array(points, (Point) {.x = l_bound + offset_x, .y = l_bound + offset_y});
    }

    return points;
}


void points_on_hull(Point_array* points, Point* p, Point* q){

    Point max;
    Point min;

    max.x = -DBL_MAX;
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


double distance(Line l, Point z){
    
    Vector v_p_z;
    Vector v_p_q;
    double t1, t2, t3, t4;

    v_p_z = init_vector(l.p, z);
    v_p_q = init_vector(l.p, l.q);

    t1 = vector_multiply(v_p_z, v_p_q);
    t2 = vector_abs(v_p_q);
    t3 = pow(t2,2);
    t4 = t1/t3;



    return vector_abs(vector_minus(v_p_z, vector_scale(v_p_q, t4)));
    
    
}


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

