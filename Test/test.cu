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



void initialData(unsigned long long int *init_array, const size_t size)
{

    for (size_t i = 0; i < size; ++i)
    {
        init_array[i] = rand()%100;
        //init_array[i] = size - 1 - i;
        //init_array[i] = i;
        //init_array[i] = 1;
    }
}



void compare_prescan_exclusive(unsigned long long int *h_data, unsigned long long int *gpuRef, size_t size)
{

    unsigned long long int* prefixSum = (unsigned long long int*)malloc(size*sizeof(unsigned long long int));
    if(!prefixSum){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    bool state = true;

    prefixSum[0] = 0;
    prefixSum[1] = h_data[0];
    // Adding present element with previous element
    for (size_t i = 2; i < size; i++)
    {
        prefixSum[i] = prefixSum[i - 1] + h_data[i-1];
    }
    for (size_t i = 0; i < size; i++)
    {
        if (prefixSum[i] != gpuRef[i])
        {
            fprintf(stdout, "Prefix sum false at index %lu: expected: %llu, actual: %llu\n", i, prefixSum[i], gpuRef[i]);
            for(size_t j = i-3; j < i+3; j++){
                fprintf(stdout, "i: %lu, h_data: %llu, prefix: %llu, gpuref: %llu\n", j, h_data[j], prefixSum[j], gpuRef[j]);
            }
            state = false;
            break;
        }
    }
    if(state){
        printf("Comparison Success\n");
    }
    else{
        printf("Comparison Failed\n");
    }
    free(prefixSum);
}

void compare_prescan_inclusive(unsigned long long int *h_data, unsigned long long int *gpuRef, size_t size)
{
    unsigned long long int* prefixSum = (unsigned long long int*)malloc(size*sizeof(unsigned long long int));
    if(!prefixSum){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    bool state = true;
    prefixSum[0] = h_data[0];
    // Adding present element with previous element
    for (size_t i = 1; i < size; i++)
    {
        prefixSum[i] = prefixSum[i-1] + h_data[i];
    }
    for (size_t i = 0; i < size; i++)
    {
        if (prefixSum[i] != gpuRef[i])
        {
            fprintf(stdout, "Prefix sum false at index %lu: expected: %llu, actual: %llu\n", i, prefixSum[i], gpuRef[i]);
            for(size_t j = i-3; j < i+3; j++){
                fprintf(stdout, "i: %lu, h_data: %llu, prefix: %llu, gpuref: %llu\n", j, h_data[j], prefixSum[j], gpuRef[j]);
            }
            state = false;
            break;
        }
    }
    if(state){
        printf("Comparison Success\n");
    }
    else{
        printf("Comparison Failed\n");
    }
    free(prefixSum);
}


void printData(unsigned long long int *data, const size_t size)
{

    fprintf(stdout, "\n");
    fprintf(stdout, "Data: ");
    for (int i = 0; i < size - 1; i++)
    {
        fprintf(stdout, "%llu, ", data[i]);
    }
    fprintf(stdout, "%llu", data[size - 1]);
    fprintf(stdout, "\n");
}


void test_sequence_5(){

    unsigned long long int* array; 
    size_t array_size;
    size_t array_bytes;
    unsigned long long int* gpuRef;

    array_size = 1000000; 
    array_bytes = array_size*(sizeof(unsigned long long int));
    array = (unsigned long long int*)malloc(sizeof(unsigned long long int)*array_size);
    if(!array){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    initialData(array, array_size);

    

    // copy back results
    gpuRef = (unsigned long long int *)malloc(array_bytes*sizeof(unsigned long long int));
    if(!gpuRef){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    memset(gpuRef, 0, array_bytes);

    master_prescan(gpuRef, array, array_size, array_bytes, EXCLUSIVE);

    // compare results
    compare_prescan_exclusive(array, gpuRef, array_size);

    free(gpuRef);
    free(array);
   
    

}

void test_max_distance_cuda(){
    int size = 10000000;

    Point_array* points = init_point_array(size);

    Point near = (Point){.x = 1, .y = 2};
    Point far = (Point){.x = 1, .y = 8};
    Line l = (Line){.p = (Point){.x = 0, .y = 0}, .q = (Point){.x = 1, .y = 1}};

    for(int i = 0; i < size; i++){
        if(i == 9000000){
            add_to_point_array(points, far);
        }else{
            add_to_point_array(points, near);
        }
    }

    Point max = max_distance_cuda(l, points);
    printf("Max dist: (%f, %f)\n", max.x, max.y);
}