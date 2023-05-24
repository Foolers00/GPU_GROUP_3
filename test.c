/////////////////////////////////////////////////////////////////////////////////////
/* TEST_PY */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef TEST
#define TEST
#include "test.h"
#endif



Point* test_sequence_1(){
    Point* p = (Point*)malloc(sizeof(Point));
    p->x = 5;
    p->y = 5;
    return p;
}


Point_array* test_sequence_2(){
    Point_array* points = NULL;
    points = init_point_array(6);

    Point u, v, w;

    u.x = 100;
    u.y = 100;

    v.x = 200;
    v.y = 200;

    w.x = 300;
    w.y = 300;

    add_to_point_array(points, u);
    add_to_point_array(points, v);
    add_to_point_array(points, w);

    print_point_array(points);

    return points;
}


Hull* test_sequence_3(){

    Hull* hull = NULL;
    hull = init_hull(6);
    Point u, v, w;

    Line l_uv;
    Line l_vw;
    Line l_uw;

    u.x = 100;
    u.y = 100;

    v.x = 200;
    v.y = 200;

    w.x = 300;
    w.y = 300;

    l_uv = init_line(u, v);
    l_vw = init_line(v, w);
    l_uw = init_line(u, w);

    add_to_hull(hull, l_uv);
    add_to_hull(hull, l_vw);
    add_to_hull(hull, l_uw);

    return hull;

}


Point_array* test_sequence_4_1(){

    Point_array* points = NULL;
    points = init_point_array(5);

    Point u, v, w, z, t;

    u.x = 0;
    u.y = 0;

    v.x = 0;
    v.y = 200;

    w.x = 100;
    w.y = 100;

    z.x = 200;
    z.y = 0;

    t.x = 200;
    t.y = 200;

    add_to_point_array(points, u);
    add_to_point_array(points, v);
    add_to_point_array(points, w);
    add_to_point_array(points, z);
    add_to_point_array(points, t);

    return points;

}


Hull* test_sequence_4_2(Point_array* points){

    return quickhull(points);

}

Point_array* test_random_generate(){
    Point_array* points = generate_random_points(50, 0, 400);
    return points;
}

Hull* test_random_hull(Point_array* points){
    return quickhull(points);
}
