/////////////////////////////////////////////////////////////////////////////////////
/* GENERAL FUNCTIONS PARALLEL */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef GENERAL_FUNCTIONS_PAR
#define GENERAL_FUNCTIONS_PAR
#include "general_functions_par.h"
#endif

#ifndef SPLIT
#define SPLIT
#include "split.h"
#endif



Hull_par* quickhull_par(Point_array_par* points){

    // device var
    int dev;

    // device set up
    dev = 0;
    CHECK(cudaSetDevice(dev));

    Line* l_pq = NULL; //device pointer
    Hull_par* hull_up = NULL;
    Hull_par* hull_down = NULL;
    Hull_par* hull_result_gpu = NULL;
    Hull_par* hull_result_cpu = NULL;

    // above below array var
    Point_array_par* points_above;
    Point_array_par* points_below;


    // set memory
    points_above = init_point_array_par_gpu(0);
    points_below = init_point_array_par_gpu(0);

    // find points on hull
    points_on_hull_par(points, &l_pq);
    //l_pq = (Line) { .p = p, .q = q };

    // splits array into above and below
    split_point_array(points, points_above, points_below, l_pq);
    

    // recursive call
    hull_up = first_quickhull_split_par(points_above, l_pq, ABOVE);
    hull_down = first_quickhull_split_par(points_below, l_pq, BELOW);

    // combine
    hull_result_gpu = combine_hull_par(hull_up, hull_down);

    // copy back results
    hull_result_cpu = init_hull_par(hull_result_gpu->size);
    CHECK(cudaMemcpy(hull_result_cpu->array, hull_result_gpu->array, hull_result_gpu->size*sizeof(Line), cudaMemcpyDeviceToHost));
    
    // free memory
    free_line_par_gpu(l_pq);
    free_hull_par_gpu(hull_result_gpu);
    free_point_array_par_gpu(points_above);
    free_point_array_par_gpu(points_below);

    return hull_result_cpu;

}

Hull_par* first_quickhull_split_par(Point_array_par* points, Line* l, int side){


    // vars
    Point_array_par* points_side = NULL;
    Line* l_p_max = NULL;
    Line* l_max_q = NULL;
    Hull_par* hull_side = NULL;

    // set memory
    points_side = points;

    // find point with max distance
    max_distance_cuda(l, points_side, &l_p_max, &l_max_q); // returns l_p_max and l_max_q gpu mem pointer, l is a device pointer
    // l_p_max = (Line) { .p = l.p, .q = max_point };
    // l_max_q = (Line) { .p = max_point, .q = l.q };


    if(points_side->size == 0) {
        hull_side = init_hull_par_gpu(1);
        CHECK(cudaMemcpy(hull_side->array , l, sizeof(Line), cudaMemcpyDeviceToDevice));
        hull_side->size = 1;
    }else if(points_side->size == 1){
        hull_side = init_hull_par_gpu(2);
        CHECK(cudaMemcpy(hull_side->array , l_p_max, sizeof(Line), cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(hull_side->array+1 , l_max_q, sizeof(Line), cudaMemcpyDeviceToDevice));
        hull_side->size = 2;

    }else {
        //points_side->curr_size > 1
        hull_side = combine_hull_par(
                quickhull_split_par(points_side, l_p_max, side),
                quickhull_split_par(points_side, l_max_q, side)
        );
    }

    // free memory
    free_line_par_gpu(l_p_max);
    free_line_par_gpu(l_max_q);

    return hull_side;

}


Hull_par* quickhull_split_par(Point_array_par* points, Line* l, int side){

    // vars
    Point_array_par* points_side = NULL;
    Line* l_p_max = NULL;
    Line* l_max_q = NULL;
    Hull_par* hull_side = NULL;

    // set memory
    points_side = init_point_array_par_gpu(0);

    // split array
    split_point_array_side(points, points_side, l, side);

    // find point with max distance
    max_distance_cuda(l, points_side, &l_p_max, &l_max_q); // returns l_p_max and l_max_q gpu mem pointer
    // l_p_max = (Line) { .p = l.p, .q = max_point };
    // l_max_q = (Line) { .p = max_point, .q = l.q };

    if(points_side->size == 0) {
        hull_side = init_hull_par_gpu(1);
        CHECK(cudaMemcpy(hull_side->array , l, sizeof(Line), cudaMemcpyDeviceToDevice));
        hull_side->size = 1;
    }else if(points_side->size == 1){
        hull_side = init_hull_par_gpu(2);
        CHECK(cudaMemcpy(hull_side->array , l_p_max, sizeof(Line), cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(hull_side->array+1 , l_max_q, sizeof(Line), cudaMemcpyDeviceToDevice));
        hull_side->size = 2;
    }else {
        //points_side->curr_size > 1
        hull_side = combine_hull_par(
                quickhull_split_par(points_side, l_p_max, side),
                quickhull_split_par(points_side, l_max_q, side)
        );
    }

    // free memory
    free_line_par_gpu(l_p_max);
    free_line_par_gpu(l_max_q);
    free_point_array_par_gpu(points_side);

    return hull_side;

}




void workload_calc(size_t* grid_size, size_t* rem_grid_size, size_t* loop_cnt, size_t* sizef, size_t size){

	size_t need_blocks;

	need_blocks = (size + 2*BLOCKSIZE-1)/(2*BLOCKSIZE);
	*loop_cnt = need_blocks/MAX_BLOCK_COUNT;
	
    *sizef = need_blocks*2*BLOCKSIZE;

	if(*loop_cnt < 1){
        *grid_size = need_blocks;
		*rem_grid_size = 0;
		*loop_cnt = 1;	
	}
	else{
        *grid_size = MAX_BLOCK_COUNT; 
		*rem_grid_size = need_blocks%MAX_BLOCK_COUNT;
	}

    

}


Point_array_par* generate_random_points_par(int num_of_points, double l_bound, double u_bound){

    time_t t;
    double difference = u_bound - l_bound;
    double offset_x = 0;
    double offset_y = 0;
    srand((unsigned) time(&t));

    Point_array_par* points = init_point_array_par(num_of_points);
    for(size_t i = 0; i < num_of_points; i++){
        offset_x = rand() % (int)difference;
        offset_y = rand() % (int)difference;
        points->array[i] = (Point) {.x = l_bound + offset_x, .y = l_bound + offset_y};
    }

    return points;
}



Point_array_par* generate_random_points_on_circle_par(int num_of_points) {
    time_t t;
    double difference = 1.0;
    double offset_x = 0.0;
    double offset_y = 0.0;
    srand((unsigned) time(&t));

    Point_array_par* points = init_point_array_par(num_of_points);
    for (size_t i = 0; i < num_of_points; i++) {
        offset_x = rand() % (int) difference;
        offset_y = rand() % (int) difference;

        double norm = sqrt(offset_x * offset_x + offset_y * offset_y);
        double normalized_x = offset_x / norm;
        double normalized_y = offset_y / norm;

        points->array[i] = (Point) {.x = normalized_x, .y = normalized_y};
    }

    return points;
}


Hull_par* combine_hull_par(Hull_par* hull_1, Hull_par* hull_2){

    // vars
    Hull_par* hull_3;
    size_t hull_1_bytes;
    size_t hull_2_bytes;
    size_t hull_3_bytes;

    // set sizes
    hull_1_bytes = hull_1->size*sizeof(Line);
    hull_2_bytes = hull_2->size*sizeof(Line);
    hull_3_bytes = hull_1_bytes+hull_2_bytes;


    // set memory
    hull_3 = init_hull_par_gpu(hull_3_bytes);
    hull_3->size = hull_1->size+hull_2->size;

    // copy results 
    CHECK(cudaMemcpy(hull_3->array, hull_1->array, hull_1_bytes, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(hull_3->array+hull_1->size, hull_2->array, hull_2_bytes, cudaMemcpyDeviceToDevice));

    // free memory
    free_hull_par_gpu(hull_1);
    free_hull_par_gpu(hull_2);

    return hull_3;

}



void points_on_hull_par(Point_array_par* points, Line** l_pq){
    minmax_cuda(points, l_pq);
}






///////////////////////////////////////////////////////////////////////////////
// Stream functions


Hull_par* quickhull_stream_par(Point_array_par* points){

    // device var
    int dev;

    // device set up
    dev = 0;
    CHECK(cudaSetDevice(dev));

    Line* l_pq = NULL; //device pointer
    Hull_par* hull_up = NULL;
    Hull_par* hull_down = NULL;
    Hull_par* hull_result_gpu = NULL;
    Hull_par* hull_result_cpu = NULL;

    // above below array var
    Point_array_par* points_above;
    Point_array_par* points_below;

    // stream var
    cudaStream_t streams[4];

    // set memory
    points_above = init_point_array_par_gpu(0);
    points_below = init_point_array_par_gpu(0);


    // init streams
    for(int i = 0; i < 4; i++){
         cudaStreamCreate(&streams[i]);
    }



    // find points on hull
    points_on_hull_stream_par(points, &l_pq, streams);
    //l_pq = (Line) { .p = p, .q = q };

    // splits array into above and below
    split_stream_point_array(points, points_above, points_below, l_pq, streams);
    

    // recursive call
    hull_up = first_quickhull_stream_split_par(points_above, l_pq, ABOVE, &streams[0]);
    hull_down = first_quickhull_stream_split_par(points_below, l_pq, BELOW, &streams[2]);

    // combine
    hull_result_gpu = combine_hull_stream_par(hull_up, hull_down, streams);

    // copy back results
    hull_result_cpu = init_hull_par(hull_result_gpu->size);
    CHECK(cudaMemcpy(hull_result_cpu->array, hull_result_gpu->array, hull_result_gpu->size*sizeof(Line), cudaMemcpyDeviceToHost));


    // destroy streams
    for(int i = 0; i < 4; i++){
         cudaStreamDestroy(streams[i]);
    }
    
    // free memory
    free_line_par_gpu(l_pq);
    free_hull_par_gpu(hull_result_gpu);
    free_point_array_par_gpu(points_above);
    free_point_array_par_gpu(points_below);

    return hull_result_cpu;

}

Hull_par* first_quickhull_stream_split_par(Point_array_par* points, Line* l, int side, cudaStream_t* streams){


    // vars
    Point_array_par* points_side = NULL;
    Line* l_p_max = NULL;
    Line* l_max_q = NULL;
    Hull_par* hull_side = NULL;

    // set memory
    points_side = points;

    // find point with max distance
    max_distance_stream_cuda(l, points_side, &l_p_max, &l_max_q, streams); // returns l_p_max and l_max_q gpu mem pointer, l is a device pointer
    // l_p_max = (Line) { .p = l.p, .q = max_point };
    // l_max_q = (Line) { .p = max_point, .q = l.q };


    if(points_side->size == 0) {
        hull_side = init_hull_par_gpu(1);
        CHECK(cudaMemcpyAsync(hull_side->array , l, sizeof(Line), cudaMemcpyDeviceToDevice, streams[0]));
        hull_side->size = 1;
    }else if(points_side->size == 1){
        hull_side = init_hull_par_gpu(2);
        CHECK(cudaMemcpyAsync(hull_side->array , l_p_max, sizeof(Line), cudaMemcpyDeviceToDevice, streams[0]));
        CHECK(cudaMemcpyAsync(hull_side->array+1 , l_max_q, sizeof(Line), cudaMemcpyDeviceToDevice, streams[1]));
        hull_side->size = 2;

    }else {
        //points_side->curr_size > 1
        hull_side = combine_hull_stream_par(
                quickhull_stream_split_par(points_side, l_p_max, side, streams),
                quickhull_stream_split_par(points_side, l_max_q, side, streams),
                streams
        );
    }

    // free memory
    free_line_par_gpu(l_p_max);
    free_line_par_gpu(l_max_q);

    return hull_side;

}


Hull_par* quickhull_stream_split_par(Point_array_par* points, Line* l, int side, cudaStream_t* streams){

    // vars
    Point_array_par* points_side = NULL;
    Line* l_p_max = NULL;
    Line* l_max_q = NULL;
    Hull_par* hull_side = NULL;

    // set memory
    points_side = init_point_array_par_gpu(0);

    // split array
    split_stream_point_array_side(points, points_side, l, side, streams);

    // find point with max distance
    max_distance_stream_cuda(l, points_side, &l_p_max, &l_max_q, streams); // returns l_p_max and l_max_q gpu mem pointer
    // l_p_max = (Line) { .p = l.p, .q = max_point };
    // l_max_q = (Line) { .p = max_point, .q = l.q };

    if(points_side->size == 0) {
        hull_side = init_hull_par_gpu(1);
        CHECK(cudaMemcpyAsync(hull_side->array , l, sizeof(Line), cudaMemcpyDeviceToDevice, streams[0]));
        hull_side->size = 1;
    }else if(points_side->size == 1){
        hull_side = init_hull_par_gpu(2);
        CHECK(cudaMemcpyAsync(hull_side->array , l_p_max, sizeof(Line), cudaMemcpyDeviceToDevice, streams[0]));
        CHECK(cudaMemcpyAsync(hull_side->array+1 , l_max_q, sizeof(Line), cudaMemcpyDeviceToDevice, streams[1]));
        hull_side->size = 2;
    }else {
        //points_side->curr_size > 1
        hull_side = combine_hull_stream_par(
                quickhull_stream_split_par(points_side, l_p_max, side, streams),
                quickhull_stream_split_par(points_side, l_max_q, side, streams),
                streams
        );
    }

    // free memory
    free_line_par_gpu(l_p_max);
    free_line_par_gpu(l_max_q);
    free_point_array_par_gpu(points_side);

    return hull_side;

}



void points_on_hull_stream_par(Point_array_par* points, Line** l_pq, cudaStream_t* streams){
    minmax_stream_cuda(points, l_pq, streams);
}




Hull_par* combine_hull_stream_par(Hull_par* hull_1, Hull_par* hull_2, cudaStream_t* streams){

    // vars
    Hull_par* hull_3;
    size_t hull_1_bytes;
    size_t hull_2_bytes;
    size_t hull_3_bytes;

    // set sizes
    hull_1_bytes = hull_1->size*sizeof(Line);
    hull_2_bytes = hull_2->size*sizeof(Line);
    hull_3_bytes = hull_1_bytes+hull_2_bytes;


    // set memory
    hull_3 = init_hull_par_gpu(hull_3_bytes);
    hull_3->size = hull_1->size+hull_2->size;

    // copy results 
    CHECK(cudaMemcpyAsync(hull_3->array, hull_1->array, hull_1_bytes, cudaMemcpyDeviceToDevice, streams[0]));
    CHECK(cudaMemcpyAsync(hull_3->array+hull_1->size, hull_2->array, hull_2_bytes, cudaMemcpyDeviceToDevice, streams[1]));

    // free memory
    free_hull_par_gpu(hull_1);
    free_hull_par_gpu(hull_2);

    return hull_3;

}