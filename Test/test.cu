/////////////////////////////////////////////////////////////////////////////////////
/* TEST_PY */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef TEST
#define TEST
#include "test.h"
#endif




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


void initialData(size_t *init_array, const size_t size)
{

    for (size_t i = 0; i < size; ++i)
    {
        init_array[i] = rand()%100;
        //init_array[i] = size - 1 - i;
        //init_array[i] = i;
        //init_array[i] = 1;
    }
}


void compare_prescan_exclusive(size_t *h_data, size_t *gpuRef, size_t size)
{

    size_t* prefixSum = (size_t*)malloc(size*sizeof(size_t));
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
            fprintf(stdout, "Prefix sum false at index %lu: expected: %lu, actual: %lu\n", i, prefixSum[i], gpuRef[i]);
            for(size_t j = i-3; j < i+3; j++){
                fprintf(stdout, "i: %lu, h_data: %lu, prefix: %lu, gpuref: %lu\n", j, h_data[j], prefixSum[j], gpuRef[j]);
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
}


void compare_prescan_inclusive(size_t *h_data, size_t *gpuRef, size_t size)
{
    size_t* prefixSum = (size_t*)malloc(size*sizeof(size_t));
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
            fprintf(stdout, "Prefix sum false at index %lu: expected: %lu, actual: %lu\n", i, prefixSum[i], gpuRef[i]);
            for(size_t j = i-3; j < i+3; j++){
                fprintf(stdout, "i: %lu, h_data: %lu, prefix: %lu, gpuref: %lu\n", j, h_data[j], prefixSum[j], gpuRef[j]);
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
}


void printData(size_t *data, const size_t size)
{

    fprintf(stdout, "\n");
    fprintf(stdout, "Data: ");
    for (int i = 0; i < size - 1; i++)
    {
        fprintf(stdout, "%lu, ", data[i]);
    }
    fprintf(stdout, "%lu", data[size - 1]);
    fprintf(stdout, "\n");
}


void test_prescan(){

    // clock
    clock_t tic = clock();
    clock_t toc = clock();
    double new_scan;
    double old_scan;
    double new_scan_avg;
    double old_scan_avg;
    int iterations = 100;

    size_t* array; 
    size_t array_size;
    size_t array_bytes;
    size_t* gpuRef;

    array_size = 10000000;

    array_bytes = array_size*(sizeof(size_t));

    #if MEMORY_MODEL == STD_MEMORY
        array = (size_t*)malloc(sizeof(size_t)*array_size);
        if(!array){
            fprintf(stderr, "Malloc failed");
            exit(1);
        }
    #elif MEMORY_MODEL == PINNED_MEMORY
        CHECK(cudaMallocHost((size_t**)&array, sizeof(size_t)*array_size));
    #elif MEMORY_MODEL == ZERO_MEMORY
        CHECK(cudaHostAlloc((void**)&array, sizeof(size_t)*array_size, cudaHostAllocMapped));
    #endif

    initialData(array, array_size);
    
    #if MEMORY_MODEL == STD_MEMORY
        gpuRef = (size_t*)malloc(array_size*sizeof(size_t));
        if(!array){
            fprintf(stderr, "Malloc failed");
            exit(1);
        }
    #else
        CHECK(cudaMallocHost((size_t**)&gpuRef, sizeof(size_t)*array_size));
    #endif
    

    memset(gpuRef, 0, array_bytes);

    for(int i = 0; i < iterations; i++){
        tic = clock();
        master_prescan(gpuRef, array, array_size, array_bytes, EXCLUSIVE);
        toc = clock();
        old_scan = (double)(toc - tic)/CLOCKS_PER_SEC;
        old_scan_avg += old_scan;

        tic = clock();
        //master_stream_prescan(gpuRef, array, array_size, array_bytes, EXCLUSIVE);
        thrust::exclusive_scan(array, array+array_size, gpuRef);
        toc = clock();
        new_scan = (double)(toc - tic)/CLOCKS_PER_SEC;
        new_scan_avg += new_scan;
    }

    old_scan_avg /= iterations;
    new_scan_avg /= iterations;

    printf("Old_scan: %f, New_scan: %f\n", old_scan, new_scan);
    //printf("Time: %f\n", new_scan);

    // compare results
    compare_prescan_exclusive(array, gpuRef, array_size);

    // free memory
    #if MEMORY_MODEL == STD_MEMORY
        free(array);
        free(gpuRef);
    #elif MEMORY_MODEL == PINNED_MEMORY || MEMORY_MODEL == ZERO_MEMORY
        CHECK(cudaFreeHost(array));
        CHECK(cudaFreeHost(gpuRef));
    #endif
   
    // reset device
    CHECK(cudaDeviceReset());

}


void test_prescan_gpu(){
    
    // device set up
    int dev; 
    dev = 0;
    CHECK(cudaSetDevice(dev));

    // clock
    clock_t tic = clock();
    clock_t toc = clock();
    double new_scan;
    double old_scan;
    double new_scan_avg;
    double old_scan_avg;
    int iterations = 1;

    size_t* array; 
    size_t array_size;
    size_t array_bytes;
    size_t* gpuRef;

    size_t array_grid_size;
    size_t array_rem_grid_size;
    size_t array_loop_cnt;
    size_t array_fsize;
    size_t array_fbytes;

    size_t* i_array_gpu;
    size_t* o_array_gpu;

    array_size = 1000;
    array_bytes = array_size*(sizeof(size_t));

    #if MEMORY_MODEL == STD_MEMORY
        array = (size_t*)malloc(sizeof(size_t)*array_size);
        if(!array){
            fprintf(stderr, "Malloc failed");
            exit(1);
        }
    #elif MEMORY_MODEL == PINNED_MEMORY
        CHECK(cudaMallocHost((size_t**)&array, sizeof(size_t)*array_size));
    #elif MEMORY_MODEL == ZERO_MEMORY
        CHECK(cudaHostAlloc((size_t**)&array, sizeof(size_t)*array_size, cudaHostAllocMapped));
    #endif

    initialData(array, array_size);

    workload_calc(&array_grid_size, &array_rem_grid_size, &array_loop_cnt,
    &array_fsize, array_size);

    array_fbytes = array_fsize*sizeof(size_t);

    #if MEMORY_MODEL == STD_MEMORY || MEMORY_MODEL == PINNED_MEMORY
        CHECK(cudaMalloc((size_t **)&i_array_gpu, array_fbytes));
    #endif


    CHECK(cudaMalloc((size_t **)&o_array_gpu, array_fbytes));
    
    #if MEMORY_MODEL == ZERO_MEMORY
        CHECK(cudaHostGetDevicePointer((void **)&i_array_gpu, (void *)array, 0));
    #else
        CHECK(cudaMemcpy(i_array_gpu, array, array_bytes, cudaMemcpyHostToDevice));
    #endif

    // copy back results
    #if MEMORY_MODEL == STD_MEMORY
    gpuRef = (size_t *)malloc(array_size*sizeof(size_t));
    if(!gpuRef){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    #else
        CHECK(cudaMallocHost((size_t**)&gpuRef, sizeof(size_t)*array_size));  
    #endif

    memset(gpuRef, 0, array_bytes);

    for(int i = 0; i < iterations; i++){
        tic = clock();
        master_prescan_gpu(o_array_gpu, i_array_gpu, array_fsize, array_fbytes, 
        array_grid_size, array_rem_grid_size, array_loop_cnt, EXCLUSIVE);
        toc = clock();
        old_scan = (double)(toc - tic)/CLOCKS_PER_SEC;
        old_scan_avg += old_scan;

        tic = clock();
        //master_stream_prescan_gpu(o_array_gpu, i_array_gpu, array_fsize, array_fbytes, 
        //array_grid_size, array_rem_grid_size, array_loop_cnt, EXCLUSIVE);
        thrust::exclusive_scan(array, array+array_size, gpuRef); //unfair
        toc = clock();
        new_scan = (double)(toc - tic)/CLOCKS_PER_SEC;
        new_scan_avg += new_scan;
    }

    old_scan_avg /= iterations;
    new_scan_avg /= iterations;

    printf("Old_scan: %f, New_scan: %f\n", old_scan, new_scan);
    //printf("Time: %f\n", new_scan);

    CHECK(cudaMemcpy(gpuRef, o_array_gpu, array_bytes, cudaMemcpyDeviceToHost));

    // compare results
    compare_prescan_exclusive(array, gpuRef, array_size);

    //free memory
    #if MEMORY_MODEL == STD_MEMORY || MEMORY_MODEL == PINNED_MEMORY
        CHECK(cudaFree(i_array_gpu));
    #endif


    CHECK(cudaFree(o_array_gpu));

    #if MEMORY_MODEL == STD_MEMORY
        free(gpuRef);
        free(array);
    #else
        CHECK(cudaFreeHost(gpuRef));
        CHECK(cudaFreeHost(array));
    #endif
    


    // reset device
    CHECK(cudaDeviceReset());

}


void test_max_distance_cuda(){
    int size = 100000000;

    Point_array_par* points = init_point_array_par(size);

    Point near = (Point){.x = 1, .y = 2};
    Point far = (Point){.x = 1, .y = 8};
    Line l = (Line){.p = (Point){.x = 0, .y = 0}, .q = (Point){.x = 1000, .y = 1000}};

    for(int i = 0; i < size; i++) {
        points->array[i] = near;
        if (i == 1234) points->array[i] = far;
    }

    Line* d_l;
    Line* l_p_max, *l_max_q;

    CHECK(cudaMalloc((void**)&d_l, sizeof(Line)));
    CHECK(cudaMemcpy(d_l, &l, sizeof(Line), cudaMemcpyHostToDevice));
    max_distance_cuda(d_l, points, &l_p_max, &l_max_q);

    Line l_p_max_host, l_max_q_host;

    CHECK(cudaMemcpy(&l_p_max_host, l_p_max, sizeof(Line), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&l_max_q_host, l_max_q, sizeof(Line), cudaMemcpyDeviceToHost));
    printf("l_p_max:\tp: (%f, %f)\tq: (%f, %f)\n", l_p_max_host.p.x, l_p_max_host.p.y, l_p_max_host.q.x, l_p_max_host.q.y);
    printf("l_max_q:\tp: (%f, %f)\tq: (%f, %f)\n", l_max_q_host.p.x, l_max_q_host.p.y, l_max_q_host.q.x, l_max_q_host.q.y);
}


void test_minmax_cuda(){
    int size = 100000000;

    Point_array_par* points = init_point_array_par(size);

    Point left = (Point){.x = -1, .y = 2};
    Point middle = (Point){.x = 100, .y = 8};
    Point right = (Point){.x = 200, .y = 3};

    for(int i = 0; i < size; i++){
        if(i == 9000000){
            points->array[i] = left;
        }else if(i == 1000000){
            points->array[i] = right;
        }else{
            points->array[i] = middle;
        }
    }

    Line* minmax;
    minmax_cuda(points, &minmax);

    Line minmax_h;
    CHECK(cudaMemcpy(&minmax_h, minmax, sizeof(Line), cudaMemcpyDeviceToHost));

    printf("minmax:\tp: (%f, %f)\tq: (%f, %f)\n", minmax_h.p.x, minmax_h.p.y, minmax_h.q.x, minmax_h.q.y);

}


void validate_minmax(){
    int max_size = 100000000;
    int iterations = 1000;
    time_t t;
    srand((unsigned) time(&t));
    for(int i = 0; i < iterations; i++){
        int size = rand() % max_size;
        int l_bound = rand() % 1000000000;
        int u_bound = rand() % 1000000000;
        Point_array* tmp = generate_random_points(size,l_bound, u_bound);
        Point_array_par* in = init_point_array_par(size);
        in->array = tmp->array;

        Point min_seq, max_seq, min_cuda, max_cuda;

        clock_t tic = clock();
        points_on_hull(tmp, &min_seq, &max_seq);
        clock_t toc = clock();
        double time_seq = (double)(toc - tic)/CLOCKS_PER_SEC;


        Line* minmax;

        tic = clock();
        minmax_cuda(in, &minmax);
        toc = clock();
        double time_cuda = (double)(toc - tic)/CLOCKS_PER_SEC;
        Line minmax_h;
        CHECK(cudaMemcpy(&minmax_h, minmax, sizeof(Line), cudaMemcpyDeviceToHost));
        min_cuda.x = minmax_h.p.x;
        min_cuda.y = minmax_h.p.y;
        max_cuda.x = minmax_h.q.x;
        max_cuda.y = minmax_h.q.y;

        printf("time seq: %f, time cuda: %f\n", time_seq, time_cuda);


        bool valid = min_seq.x == min_cuda.x &&  max_seq.x == max_cuda.x;
        if(valid){
            printf("no error found so far\n size: %i, l_bound: %i, u_bound: %i. Min seq: [%f, %f], Min cuda: [%f, %f], Max seq: [%f, %f], Max cuda: [%f, %f]\n", size, l_bound, u_bound, min_seq.x, min_seq.y, min_cuda.x, min_cuda.y, max_seq.x, max_seq.y, max_cuda.x, max_cuda.y);
        }else {
            printf("found error.\n size: %i, l_bound: %i, u_bound: %i. Min seq: [%f, %f], Min cuda: [%f, %f], Max seq: [%f, %f], Max cuda: [%f, %f]\n", size, l_bound, u_bound, min_seq.x, min_seq.y, min_cuda.x, min_cuda.y, max_seq.x, max_seq.y, max_cuda.x, max_cuda.y);
            exit(1);
        }
    }
}


void test_split(){

    // device var
    int dev;

    // device set up
    dev = 0;
    CHECK(cudaSetDevice(dev));

    // clock
    clock_t tic;
    clock_t toc;
    double cpu_time;
    double gpu_time;
    double thrust_time;

    // main array var
    int size;
    int l_bound;
    int u_bound;

    // point arrays cpu var
    Point_array* points_cpu;
    Point_array* points_above_cpu;
    Point_array* points_below_cpu;

    // point array gpu var
    Point_array_par* points_gpu;
    Point_array_par* points_above_gpu;
    Point_array_par* points_below_gpu;
    Point_array_par* temp_above;
    Point_array_par* temp_below;

    // thrust var
    thrust::device_vector<Point> points_thrust; 
    thrust::device_vector<Point> points_above_thrust; 
    thrust::device_vector<Point> points_below_thrust;
    thrust::device_vector<Line> l_pq_thrust;
    std::vector<Point> points_temp_above_thrust; 
    std::vector<Point> points_temp_below_thrust;

    // point on hull var
    Point p;
    Point q;
    Line l_pq;
    Line* l_pq_gpu;

    // state var for compare
    bool state = true;

    // set up array
    size = 1000000;
    l_bound = 0;
    u_bound = 100000000;

    points_cpu = generate_random_points(size, l_bound, u_bound);
    points_gpu = init_point_array_par(size);
    points_thrust.resize(size);

    // copy memory to gpu
    memcpy(points_gpu->array, points_cpu->array, sizeof(Point)*size);

    // copy memory to host thrust
    thrust::copy(&points_cpu->array[0], &points_cpu->array[size], points_thrust.begin());
     


    tic = clock();
    // init above/below arrays
    points_above_cpu = init_point_array(points_cpu->max_size/2);
    points_below_cpu = init_point_array(points_cpu->max_size/2);

    // points on hull
    //points_on_hull(points_cpu, &p, &q);
    p.x = 79.000000;
    p.y = 952.000000;
    q.x = 140.000000;
    q.y = 332.000000;
    l_pq = (Line) { .p = p, .q = q };


    // CPU Version

    for(size_t i = 0; i < points_cpu->curr_size; i++){
        int result = check_point_location(l_pq, points_cpu->array[i]);
        if(result == ON){
            continue;
        }

        if(result == ABOVE){
            add_to_point_array(points_above_cpu, points_cpu->array[i]);
        }
        else{
            add_to_point_array(points_below_cpu, points_cpu->array[i]);
        }
    }
    toc = clock();
    cpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;

    // GPU Version


    points_above_gpu = init_point_array_par_gpu(0);
    points_below_gpu = init_point_array_par_gpu(0);

    CHECK(cudaMalloc((Line **)&l_pq_gpu, sizeof(Line)));
    CHECK(cudaMemcpy(l_pq_gpu, &l_pq, sizeof(Line), cudaMemcpyHostToDevice));


    tic = clock();
    // splits array into above and below

    //split_point_array(points_gpu, points_above_gpu, points_below_gpu, l_pq_gpu);
    Point_array_par* points_gpu_gpu = init_point_array_par_gpu(points_gpu->size);
    CHECK(cudaMemcpy(points_gpu_gpu->array, points_gpu->array, points_gpu->size*sizeof(Point), cudaMemcpyHostToDevice));
    split_point_array_side(points_gpu_gpu, points_above_gpu, l_pq_gpu, ABOVE);
    split_point_array_side(points_gpu_gpu, points_below_gpu, l_pq_gpu, BELOW);

    toc = clock();
    gpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;


    // THRUST Version
    tic = clock();

    l_pq_thrust.insert(l_pq_thrust.end(), l_pq);
    thrust_split_point_array(points_thrust, points_above_thrust, points_below_thrust, l_pq_thrust);

    toc = clock();
    thrust_time = (double)(toc - tic)/CLOCKS_PER_SEC;

    // copy back results
    temp_above = init_point_array_par(points_above_gpu->size*sizeof(Point));
    temp_below = init_point_array_par(points_below_gpu->size*sizeof(Point));

    CHECK(cudaMemcpy(temp_above->array, points_above_gpu->array, points_above_gpu->size*sizeof(Point), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(temp_below->array, points_below_gpu->array, points_below_gpu->size*sizeof(Point), cudaMemcpyDeviceToHost));
    
    points_temp_above_thrust.resize(points_above_thrust.size());
    points_temp_below_thrust.resize(points_below_thrust.size());
    thrust::copy(points_above_thrust.begin(), points_above_thrust.end(), points_temp_above_thrust.begin());
    thrust::copy(points_below_thrust.begin(), points_below_thrust.end(), points_temp_below_thrust.begin());


    // compare results
    printf("Above results: ");
    for(size_t i = 0; i < points_above_cpu->curr_size; i++){
        if(!compare_points(points_above_cpu->array[i], temp_above->array[i])){

            printf("x or y are not the same: x: %f, %f, y: %f, %f\n",
            points_above_cpu->array[i].x, temp_above->array[i].x,
            points_above_cpu->array[i].y, temp_above->array[i].y);

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

    state = true;

    printf("Above results thrust: ");
    for(size_t i = 0; i < points_above_cpu->curr_size; i++){
        if(!compare_points(points_above_cpu->array[i], points_temp_above_thrust[i])){

            printf("x or y are not the same: x: %f, %f, y: %f, %f\n",
            points_above_cpu->array[i].x, points_temp_above_thrust[i].x,
            points_above_cpu->array[i].y, points_temp_above_thrust[i].y);

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

    state = true;

    printf("Below results: ");
    for(size_t i = 0; i < points_below_cpu->curr_size; i++){
        if(!compare_points(points_below_cpu->array[i], temp_below->array[i])){
            printf("x or y are not the same: x: %f, %f, y: %f, %f\n",
            points_below_cpu->array[i].x, temp_below->array[i].x,
            points_below_cpu->array[i].y, temp_below->array[i].y);

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

    state = true;

    printf("Below results thrust: ");
    for(size_t i = 0; i < points_below_cpu->curr_size; i++){
        if(!compare_points(points_below_cpu->array[i], points_temp_below_thrust[i])){
            printf("x or y are not the same: x: %f, %f, y: %f, %f\n",
            points_below_cpu->array[i].x, points_temp_below_thrust[i].x,
            points_below_cpu->array[i].y, points_temp_below_thrust[i].y);

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

    state = true;

    printf("Size results: ");
    if(points_above_cpu->curr_size != points_above_gpu->size ||
        points_below_cpu->curr_size != points_below_gpu->size){
            printf("Sizes do not match: Above: %lu, %lu, Below: %lu, %lu\n",
            points_above_cpu->curr_size, points_above_gpu->size, points_below_cpu->curr_size,
            points_below_gpu->size);
            state = false;
    }

    if(state){
        printf("Comparison Success\n");
    }
    else{
       printf("Comparison Failed\n"); 
    }

    state = true;

    printf("Size results thurst: ");
    if(points_above_cpu->curr_size != points_temp_above_thrust.size() ||
        points_below_cpu->curr_size != points_temp_below_thrust.size()){
            printf("Sizes do not match: Above: %lu, %lu, Below: %lu, %lu\n",
            points_above_cpu->curr_size, points_temp_above_thrust.size(), points_below_cpu->curr_size,
            points_temp_below_thrust.size());
            state = false;
    }

    if(state){
        printf("Comparison Success\n");
    }
    else{
       printf("Comparison Failed\n"); 
    }


    if(state){
        printf("Comparison Success\n");
        printf("CPU time: %f, GPU time: %f, Thrust time: %f\n", cpu_time, gpu_time, thrust_time);
    }


    // free memory
    free(points_cpu);
    free(points_above_cpu);
    free(points_below_cpu);

    free_point_array_par(points_gpu);
    free_point_array_par_gpu(points_above_gpu);
    free_point_array_par_gpu(points_below_gpu);
    free_point_array_par(temp_above);
    free_point_array_par(temp_below);

    CHECK(cudaFree(l_pq_gpu));

    // reset device
    //CHECK(cudaDeviceReset());

}


void test_combinehull(){

    // device var
    int dev;

    // device set up
    dev = 0;
    CHECK(cudaSetDevice(dev));

    // vars
    size_t hull_1_size;
    size_t hull_1_bytes;

    size_t hull_2_size;
    size_t hull_2_bytes;

    size_t hull_3_bytes;

    int l_bound;
    int u_bound;

    Hull* hull_1_cpu;
    Hull* hull_2_cpu;
    Hull* hull_3_cpu;

    Hull_par* hull_1_gpu;
    Hull_par* hull_2_gpu;
    Hull_par* hull_3_gpu;

    Line* temp;

    bool state = true;

    // set vars
    hull_1_size = 10;
    hull_1_bytes = sizeof(Line)*hull_1_size;
    hull_2_size = 10;
    hull_2_bytes = sizeof(Line)*hull_2_size;
    l_bound = 0;
    u_bound = rand()%100000;

    hull_1_cpu = generate_random_lines(hull_1_size, l_bound, u_bound);
    hull_2_cpu = generate_random_lines(hull_2_size, l_bound, u_bound);

    hull_1_gpu = init_hull_par_gpu(hull_1_size);
    hull_2_gpu = init_hull_par_gpu(hull_2_size);

    hull_1_gpu->size = hull_1_size;
    hull_2_gpu->size = hull_2_size;


    CHECK(cudaMemcpy(hull_1_gpu->array , hull_1_cpu->array, hull_1_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(hull_2_gpu->array , hull_2_cpu->array, hull_2_bytes, cudaMemcpyHostToDevice));

    hull_3_cpu = combine_hull(hull_1_cpu, hull_2_cpu);
    hull_3_gpu = combine_hull_par(hull_1_gpu, hull_2_gpu);

    hull_3_bytes = hull_3_gpu->size*sizeof(Line);

    temp = (Line*)malloc(hull_3_bytes);
    if(!temp){
        fprintf(stderr, "Malloc failed");
        exit(1);       
    }

    CHECK(cudaMemcpy(temp , hull_3_gpu->array, hull_3_bytes, cudaMemcpyDeviceToHost));

    printf("Combine result: ");
    for(int i = 0; i < hull_3_cpu->curr_size; i++){
        if(!compare_lines(hull_3_cpu->array[i], temp[i])){
            printf("Lines do not match: Point 1 cpu: (%f, %f),Point 2 cpu: (%f, %f), Point 1 gpu: (%f, %f), Point 2 gpu: (%f, %f)\n", 
            hull_3_cpu->array[i].p.x,
            hull_3_cpu->array[i].p.y,
            hull_3_cpu->array[i].q.x,
            hull_3_cpu->array[i].q.y,
            temp[i].p.x,
            temp[i].p.y,
            temp[i].q.x,
            temp[i].q.y);
            state = false;
        }
    }

    if(state){
        printf("Comparison Success\n");
    }
    else{
       printf("Comparison Failed\n"); 
    }

    printf("Size result: " );
    if(hull_3_cpu->curr_size != hull_3_gpu->size){
        printf("Sizes do not match: %lu, %lu\n", hull_3_cpu->curr_size, hull_3_gpu->size);
        state = false;
    }

    if(state){
        printf("Comparison Success\n");
    }
    else{
       printf("Comparison Failed\n"); 
    }

    // free memory
    free_hull(hull_3_cpu);
    free(temp);
    free_hull_par_gpu(hull_3_gpu);

    // reset device
    CHECK(cudaDeviceReset());

}




void test_quickhull(){

    // device var
    int dev;

    // device set up
    dev = 0;
    CHECK(cudaSetDevice(dev));

    // clock
    clock_t tic;
    clock_t toc;
    double cpu_time;
    double gpu_time;

    // vars
    size_t size;
    double l_bound;
    double u_bound;

    // cpu
    Point_array* points_cpu;
    Hull* hull_cpu;

    // gpu
    Point_array_par* points_gpu;
    Hull_par* hull_gpu;

    // state
    bool state = true;


    // set vars
    size = 100000000;
    l_bound = -10000000;
    u_bound = 10000000;

    
    points_cpu = init_point_array(2*size);
    points_gpu = generate_random_points_par(size, l_bound, u_bound);
    //points_gpu = generate_random_points_on_circle_par(size, 10);

    //readPointsFromCSV("points", &points_gpu);


    memcpy(points_cpu->array, points_gpu->array, points_gpu->size*sizeof(Point));
    points_cpu->curr_size = size;

    //print_point_array(points_cpu);

    //writePointArrayToCSV(points_cpu);

    tic = clock();
    hull_cpu = quickhull(points_cpu);
    toc = clock();
    cpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;

    printf("CPU Finished\n");

    //writeHullArrayToCSV(hull_cpu);

    tic = clock();
    hull_gpu = quickhull_stream_par(points_gpu);
    toc = clock();
    gpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;

    printf("GPU Finished\n");

    //writeHullparArrayToCSV(hull_gpu);


    bool state_2 = false;

    printf("Compare result: ");
    for(int i = 0; i < hull_cpu->curr_size; i++){
        state_2 = false;
        for(int j = 0; j < hull_gpu->size; j++){
            if(compare_lines(hull_cpu->array[i], hull_gpu->array[j])){
                state_2 = true;
            }
            else{
                if(hull_cpu->array[i].p.x == hull_cpu->array[i].q.x && hull_gpu->array[j].p.x == hull_gpu->array[j].q.x &&
                    hull_cpu->array[i].p.x == hull_gpu->array[j].p.x){
                    state_2 = true;
                }
                if(hull_cpu->array[i].p.y == hull_cpu->array[i].q.y && hull_gpu->array[j].p.y == hull_gpu->array[j].q.y &&
                    hull_cpu->array[i].p.y == hull_gpu->array[j].p.y){
                    state_2 = true;
                }
            }

        }
        if(state_2 == false){
            state = false;
            printf("This lines does not appear: Cpu: (%f, %f)-(%f, %f)\n",
                    hull_cpu->array[i].p.x, hull_cpu->array[i].p.y, hull_cpu->array[i].q.x, hull_cpu->array[i].q.y);
            for(int z = 0; z < hull_gpu->size; z++){
                printf("Gpu lines: (%f, %f)-(%f, %f)\n",  hull_gpu->array[z].p.x, hull_gpu->array[z].p.y, hull_gpu->array[z].q.x, hull_gpu->array[z].q.y);
            }
        }
        
    }


    if(state){
        printf("Comparison Success\n");
    }
    else{
       printf("Comparison Failed\n"); 
    }


    printf("Size result: ");
    if(points_cpu->curr_size != points_gpu->size){
        printf("Sizes do not match: CPU: %lu, GPU: %lu\n", points_cpu->curr_size, points_gpu->size);
        state = false;
    }

    if(state){
        printf("Comparison Success\n");
    }
    else{
       printf("Comparison Failed\n"); 
    }


    if(state){
        printf("Comparison Success\n");
        printf("CPU time: %f, GPU time: %f\n", cpu_time, gpu_time);
    }
   

    // free memory
    free_point_array(points_cpu);
    free_point_array_par(points_gpu);
    free_hull(hull_cpu);
    free_hull_par(hull_gpu);


    //}

    // reset device
    CHECK(cudaDeviceReset());
}







void test_thrust_quickhull(){

    // device var
    int dev;

    // device set up
    dev = 0;
    CHECK(cudaSetDevice(dev));

    // clock
    clock_t tic;
    clock_t toc;
    double cpu_time;
    double gpu_time;

    // vars
    size_t size;
    double l_bound;
    double u_bound;

    // cpu
    Point_array* points_cpu;
    Hull* hull_cpu;

    // gpu
    Point_array_par* points_gpu;
    Hull_par* hull_gpu;

    // state
    bool state = true;


    // set vars
    size = 10000;
    l_bound = 0;
    u_bound = 1000;

    points_cpu = init_point_array(2*size);
    points_gpu = generate_random_points_par(size, l_bound, u_bound);

    //readPointsFromCSV("points", &points_gpu);


    memcpy(points_cpu->array, points_gpu->array, points_gpu->size*sizeof(Point));
    points_cpu->curr_size = size;

    //writePointArrayToCSV(points_cpu);

    tic = clock();
    hull_cpu = quickhull(points_cpu);
    toc = clock();
    cpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;

    //writeHullArrayToCSV(hull_cpu);

    thrust::host_vector<Line> hull_gpu_vec;
    tic = clock();
    thrust_quickhull(points_gpu, hull_gpu_vec);
    toc = clock();
    gpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;

    hull_gpu = init_hull_par(hull_gpu_vec.size());
    for (int i = 0; i < hull_gpu_vec.size(); i++){
        hull_gpu->array[i] = hull_gpu_vec[i];
    }

    //writeHullparArrayToCSV(hull_gpu);

    bool state_2 = false;

    printf("Compare result: ");
    for(int i = 0; i < hull_cpu->curr_size; i++){
        state_2 = false;
        for(int j = 0; j < hull_gpu->size; j++){
            if(compare_lines(hull_cpu->array[i], hull_gpu->array[j])){
                state_2 = true;
            }
            else{
                if(hull_cpu->array[i].p.x == hull_cpu->array[i].q.x && hull_gpu->array[j].p.x == hull_gpu->array[j].q.x &&
                   hull_cpu->array[i].p.x == hull_gpu->array[j].p.x){
                    state_2 = true;
                }
                if(hull_cpu->array[i].p.y == hull_cpu->array[i].q.y && hull_gpu->array[j].p.y == hull_gpu->array[j].q.y &&
                   hull_cpu->array[i].p.y == hull_gpu->array[j].p.y){
                    state_2 = true;
                }
            }

        }
        if(state_2 == false){
            state = false;
            printf("This lines does not appear: Cpu: (%f, %f)-(%f, %f)\n",
                   hull_cpu->array[i].p.x, hull_cpu->array[i].p.y, hull_cpu->array[i].q.x, hull_cpu->array[i].q.y);
            for(int z = 0; z < hull_gpu->size; z++){
                printf("Gpu lines: (%f, %f)-(%f, %f)\n",  hull_gpu->array[z].p.x, hull_gpu->array[z].p.y, hull_gpu->array[z].q.x, hull_gpu->array[z].q.y);
            }
        }

    }


    if(state){
        printf("Comparison Success\n");
    }
    else{
        printf("Comparison Failed\n");
    }


    printf("Size result: ");
    if(points_cpu->curr_size != points_gpu->size){
        printf("Sizes do not match: CPU: %lu, GPU: %lu\n", points_cpu->curr_size, points_gpu->size);
        state = false;
    }

    if(state){
        printf("Comparison Success\n");
    }
    else{
        printf("Comparison Failed\n");
    }


    if(state){
        printf("Comparison Success\n");
        printf("CPU time: %f, GPU time: %f\n", cpu_time, gpu_time);
    }


    // free memory
    free_point_array(points_cpu);
    free_point_array_par(points_gpu);
    free_hull(hull_cpu);
    free_hull_par(hull_gpu);

    //}

    // reset device
    CHECK(cudaDeviceReset());
}




void test_quickhull_performance(){

    // device var
    int dev;

    // device set up
    dev = 0;
    CHECK(cudaSetDevice(dev));

    // clock
    clock_t tic;
    clock_t toc;
    double cpu_time;
    double gpu_time;
    double thrust_gpu_time;
    double cpu_time_avg;
    double gpu_time_avg;
    double thrust_gpu_time_avg;
    int iterations;

    // vars
    size_t size;
    double l_bound;
    double u_bound;

    // cpu
    Point_array* points_cpu;
    Hull* hull_cpu;

    // gpu / thrust
    Point_array_par* points_gpu;
    Hull_par* hull_gpu;


    // set vars
    size = 100000;
    l_bound = INT_MIN;
    u_bound = INT_MAX;

    cpu_time_avg = 0;
    gpu_time_avg = 0;
    thrust_gpu_time_avg = 0;
    iterations = 0;

    while(iterations < 3){

        points_cpu = init_point_array(2*size);
        //points_gpu = generate_random_points_par(size, l_bound, u_bound);
        points_gpu = generate_random_points_on_circle_par(size, 1000);


        memcpy(points_cpu->array, points_gpu->array, points_gpu->size*sizeof(Point));
        points_cpu->curr_size = size;

        // cpu
        tic = clock();
        hull_cpu = quickhull(points_cpu);
        toc = clock();
        cpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;

        // gpu
        tic = clock();
        hull_gpu = quickhull_par(points_gpu);
        toc = clock();
        gpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;

        // thrust
        thrust::host_vector<Line> hull_gpu_vec;
        tic = clock();
        thrust_quickhull(points_gpu, hull_gpu_vec);
        toc = clock();
        thrust_gpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;

        printf("CPU Size: %lu, GPU Size: %lu, Thrust Size: %lu\n", hull_cpu->curr_size, hull_gpu->size, hull_gpu_vec.size());


        // free memory
        free_point_array(points_cpu);
        free_point_array_par(points_gpu);
        free_hull(hull_cpu);
        free_hull_par(hull_gpu);

        cpu_time_avg += cpu_time;
        gpu_time_avg += gpu_time;
        thrust_gpu_time_avg += thrust_gpu_time;
        iterations++;

    }

    cpu_time_avg/=iterations;
    gpu_time_avg/=iterations;
    thrust_gpu_time_avg/=iterations;
    
    printf("CPU time: %f, GPU time: %f, GPU Thrust time: %f\n", cpu_time_avg, 
    gpu_time_avg, thrust_gpu_time_avg);
    
    // reset device
    CHECK(cudaDeviceReset());
}

void test_quickhull_performance(size_t size, FILE* output_file){
    // device var
    int dev;

    // device set up
    dev = 0;
    CHECK(cudaSetDevice(dev));

    // clock
    clock_t tic;
    clock_t toc;
    double cpu_time;
    double gpu_time;
    double thrust_gpu_time;
    double cpu_time_avg;
    double gpu_time_avg;
    double thrust_gpu_time_avg;
    int iterations;

    // vars
    double l_bound;
    double u_bound;

    // cpu
    Point_array* points_cpu;
    Hull* hull_cpu;

    // gpu / thrust
    Point_array_par* points_gpu;
    Hull_par* hull_gpu;


    // set vars
    l_bound = -1;
    u_bound = 1;

    cpu_time_avg = 0;
    gpu_time_avg = 0;
    thrust_gpu_time_avg = 0;
    iterations = 0;

    while(iterations < 100){

        points_cpu = init_point_array(2*size);
        points_gpu = generate_random_points_par(size, l_bound, u_bound);


        memcpy(points_cpu->array, points_gpu->array, points_gpu->size*sizeof(Point));
        points_cpu->curr_size = size;

        // cpu
        tic = clock();
        hull_cpu = quickhull(points_cpu);
        toc = clock();
        cpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;

        // gpu
        tic = clock();
        hull_gpu = quickhull_par(points_gpu);
        toc = clock();
        gpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;

        // thrust
        thrust::host_vector<Line> hull_gpu_vec;
        tic = clock();
        thrust_quickhull(points_gpu, hull_gpu_vec);
        toc = clock();
        thrust_gpu_time = (double)(toc - tic)/CLOCKS_PER_SEC;


        // free memory
        free_point_array(points_cpu);
        free_point_array_par(points_gpu);
        free_hull(hull_cpu);
        free_hull_par(hull_gpu);

        cpu_time_avg += cpu_time;
        gpu_time_avg += gpu_time;
        thrust_gpu_time_avg += thrust_gpu_time;
        iterations++;

    }

    cpu_time_avg/=iterations;
    gpu_time_avg/=iterations;
    thrust_gpu_time_avg/=iterations;
    
    // printf("CPU time: %f, GPU time: %f, GPU Thrust time: %f\n", cpu_time_avg, 
    // gpu_time_avg, thrust_gpu_time_avg);
    fprintf(output_file, "CPU time: %f, GPU time: %f, GPU Thrust time: %f, MEMORY_MODEL: %d, BLOCKSIZE: %d, size: %zu\n",
        cpu_time_avg, gpu_time_avg, thrust_gpu_time_avg, MEMORY_MODEL, BLOCKSIZE, size);



    // reset device
    CHECK(cudaDeviceReset());
}



void writePointArrayToCSV(Point_array* points){
    FILE *file = fopen("points", "w");
    if (file == NULL) {
        printf("Error opening file: %s\n", "points");
        return;
    }

    for (int i = 0; i < points->curr_size; i++) {
        fprintf(file, "%f,%f\n", points->array[i].x,points->array[i].y);
    }

    fclose(file);
    printf("Array successfully written to CSV file: %s\n", "points");
}


void writeHullArrayToCSV(Hull* hull) {
    FILE *file = fopen("cpu_hull", "w");
    if (file == NULL) {
        printf("Error opening file: %s\n", "cpu_hull");
        return;
    }

    for (int i = 0; i < hull->curr_size; i++) {
        fprintf(file, "%f,%f,%f,%f\n", hull->array[i].p.x,hull->array[i].p.y,hull->array[i].q.x,hull->array[i].q.y);
    }

    fclose(file);
    printf("Array successfully written to CSV file: %s\n", "cpu_hull");
}


void writeHullparArrayToCSV(Hull_par* hull) {
    FILE *file = fopen("gpu_hull", "w");
    if (file == NULL) {
        printf("Error opening file: %s\n", "gpu_hull");
        return;
    }

    for (int i = 0; i < hull->size; i++) {
        fprintf(file, "%f,%f,%f,%f\n", hull->array[i].p.x,hull->array[i].p.y,hull->array[i].q.x,hull->array[i].q.y);
    }

    fclose(file);
    printf("Array successfully written to CSV file: %s\n", "gpu_hull");
}





void readPointsFromCSV(const char* filename, Point_array_par** points) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
    }

    // Count the number of lines in the file
    int lines = 0;
    char ch;
    while (!feof(file)) {
        ch = fgetc(file);
        if (ch == '\n') {
            lines++;
        }
    }
    rewind(file);  // Reset the file pointer to the beginning

    // Allocate memory for the points
    *points = (Point_array_par*)malloc(sizeof(Point_array_par));
    if(!points){
        printf("Malloc failed");
    }

    (*points)->array = (Point*)malloc(lines * sizeof(Point));
    if(!(*points)->array){
        printf("Malloc failed");
    }

    // Read points from the file
    int i = 0;
    while (fscanf(file, "%lf,%lf", &(*points)->array[i].x, &(*points)->array[i].y) != EOF) {
        i++;
    }

    fclose(file);
    (*points)->size = lines;
    printf("Points successfully read from CSV file: %s\n", filename);
}




Hull* generate_random_lines(int num_of_lines, double l_bound, double u_bound){

    time_t t;
    double difference = u_bound - l_bound;
    double offset_x_1 = 0;
    double offset_y_1 = 0;
    double offset_x_2 = 0;
    double offset_y_2 = 0;
    Point point_1;
    Point point_2;
    Line l;
    srand((unsigned) time(&t));

    Hull* hull = init_hull(num_of_lines * 2);
    for(size_t i = 0; i < num_of_lines; i++){
        offset_x_1 = rand() % (int)difference;
        offset_y_1 = rand() % (int)difference;
        offset_x_2 = rand() % (int)difference;
        offset_y_2 = rand() % (int)difference;
        point_1 = (Point) {.x = l_bound + offset_x_1, .y = l_bound + offset_y_1};
        point_2 = (Point) {.x = l_bound + offset_x_2, .y = l_bound + offset_y_2};
        l = init_line(point_1, point_2);
        add_to_hull(hull, l);
    }

    return hull;
}



void test_memory_model() {

    // // Set up different memory models to test
    // int memory_models[] = {STD_MEMORY, PINNED_MEMORY, ZERO_MEMORY};
    // int num_models = sizeof(memory_models) / sizeof(memory_models[0]);

    // Print the value of BLOCKSIZE
    std::cout << "BLOCKSIZE: " << BLOCKSIZE << std::endl;

    // Define the maximum input size
    int max_input_size = 1000000;

    std::string fileSuffix = "_" + std::to_string(MEMORY_MODEL) + "_" + std::to_string(BLOCKSIZE);
    std::string fileName = "test_memory_output" + fileSuffix + ".txt";
    
    FILE* output_file = fopen(fileName.c_str(), "w");    
    if (output_file == NULL) {
        printf("Error opening output file.\n");
        return;
    }

    // for (int i = 0; i < num_models; i++) {
    //     // Set MEMORY_MODEL to current memory model
    //     #undef MEMORY_MODEL
    //     #define MEMORY_MODEL memory_models[i]

    //     // Perform testing with current memory model
    //     printf("Testing with MEMORY_MODEL: %d\n", MEMORY_MODEL);

    for (int size = 10; size <= max_input_size; size *= 10) {
        printf("Testing with input size: %d\n", size);

        // Call the function and capture the output
        test_quickhull_performance(size, output_file);

        // fprintf(output_file, "CPU time: %f, GPU time: %f, GPU Thrust time: %f, MEMORY_MODEL: %d, BLOCKSIZE: %d, size: %zu\n",
        //         cpu_time_avg, gpu_time_avg, thrust_gpu_time_avg, MEMORY_MODEL, BLOCKSIZE, size);
    
    }

    fclose(output_file);  // Close the output file
}
