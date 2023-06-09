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
    Point_array* points = generate_random_points(15, -200, 800);
    return points;
}


Hull* test_random_hull(Point_array* points){
    return quickhull(points);
}


void test_new_quickhull(){

    Point_array* points;
    Hull* old_hull;
    Hull* new_hull;

    clock_t tic;
    clock_t toc;
    double old_time;
    double new_time;
    double old_avg = 0;
    double new_avg = 0;
    int iterations = 1; 

    int size = 100000000;
    int l_bound = 0;
    int u_bound = 10000;

    bool state = true;

    points = generate_random_points(size, l_bound, u_bound);

    for(int i = 0; i < iterations; i++){
        tic = clock();
        old_hull = quickhull(points);
        toc = clock();

        old_time = (double)(toc - tic)/CLOCKS_PER_SEC;

        old_avg += old_time;

        tic = clock();
        new_hull = new_quickhull(points);
        toc = clock();

        new_time = (double)(toc - tic)/CLOCKS_PER_SEC;

        new_avg += new_time;
    }

    old_avg /= iterations;
    new_avg /= iterations;

    printf("Comparison result: ");
    for(int i = 0; i < old_hull->curr_size; i++){
        if(!compare_lines(old_hull->array[i], new_hull->array[i])){
            printf("Lines do not match\n");
            state = false;
            break;
        }
    }

    if(state){
        printf("Comparison Success\n");
        printf("Old time: %f, New time: %f\n", old_avg, new_avg);
    }
    else{
       printf("Comparison Failed\n"); 
    }

}
