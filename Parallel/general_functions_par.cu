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

    Point p;
    Point q;
    Line l_pq;
    Hull_par* hull_up = NULL;
    Hull_par* hull_down = NULL;

    // above below array var
    Point_array_par* points_above;
    Point_array_par* points_below;

    // points on hull
    ///////////////////////////////////

    // splits array into above and below
    split_point_array(points, points_above, points_below, l_pq);

    // hull_up = quickhull_split_par(&points_above, l_pq, ABOVE);
    // hull_down = quickhull_split_par(&points_below, l_pq, BELOW);

    return combine_hull_par(hull_up, hull_down);

}


Hull_par* quickhull_split_par(Point_array_par* points, Line l, int side){

    Point_array_par* points_side = NULL;
    Point max_point;
    Line l_p_max;
    Line l_max_q;
    Hull_par* hull_side = NULL;

    split_point_array_side(points, points_side, l, side);


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



Hull_par* combine_hull_par(Hull_par* hull_1, Hull_par* hull_2){

    // Hull* new_hull = (Hull*)malloc(sizeof(Hull));

    // new_hull->curr_size = hull_1->curr_size+hull_2->curr_size;

    // if(hull_1->max_size >= hull_2->max_size){
    //     new_hull->max_size = hull_1->max_size<<2;
    // }
    // else{
    //     new_hull->max_size = hull_2->max_size<<2;
    // }

    // new_hull->array = (Line*)malloc(new_hull->max_size*sizeof(Line));

    // memcpy(new_hull->array, hull_1->array, hull_1->curr_size*sizeof(Line));
    // memcpy(&new_hull->array[hull_1->curr_size], hull_2->array, hull_2->curr_size*sizeof(Line));

    // free_hull(hull_1);
    // free_hull(hull_2);

    // return new_hull;

}

