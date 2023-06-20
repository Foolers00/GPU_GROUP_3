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

    hull_up = quickhull_split(points, l_pq, ABOVE);
    hull_down = quickhull_split(points, l_pq, BELOW);

    return combine_hull(hull_up, hull_down);

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

        if(side == result){
            add_to_point_array(points_side, points->array[i]);
        }
    }

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

    if(result>0){
        return ABOVE;
    }
    if(result == 0){
        return ON;
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


Point_array* generate_random_points_on_circle(int num_of_points){

    time_t t;
    double difference = 2.0;
    double offset_x = 0.0;
    double offset_y = 0.0;
    srand((unsigned) time(&t));

    Point_array* points = init_point_array(num_of_points * 2);
    for(size_t i = 0; i < num_of_points; i++){        
        offset_x = rand() % (int)difference;
        offset_y = rand() % (int)difference;
        double norm = sqrt(offset_x * offset_x + offset_y * offset_y);
        add_to_point_array(points, (Point) {.x = offset_x / norm, .y = offset_y / norm});
    }

    return points;
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

