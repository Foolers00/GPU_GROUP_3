/////////////////////////////////////////////////////////////////////////////////////
/* TEST_PY */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef TEST
#define TEST
#include "test.h"
#endif




Point_array* test_sequence_1(){
    Point_array* points = (Point_array*)malloc(sizeof(Point_array));
    init_point_array(points, 6);

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

    return points;
}


Point* test_sequence_2(){
    Point* p = (Point*)malloc(sizeof(Point));
    p->x = 5;
    p->y = 5;
    return p;
}