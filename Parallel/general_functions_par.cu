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
    Point_array_par points_above;
    Point_array_par points_below;

    // points on hull
    ///////////////////////////////////

    // splits array into above and below
    split_point_array(points, &points_above, &points_below, l_pq);

    hull_up = quickhull_split_par(&points_above, l_pq, ABOVE);
    hull_down = quickhull_split_par(&points_below, l_pq, BELOW);

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



int* workload_calc(int* grid_size, size_t* array_fsize, size_t array_size){

    int need_blocks;
    int load_per_block;
    int rem_blocks;
    int* workload; 

    need_blocks =  (array_size + 2*BLOCKSIZE - 1) / (2*BLOCKSIZE);
    
    if(array_fsize){
        *array_fsize = need_blocks*2*BLOCKSIZE; 
    }

    if(need_blocks>=MAX_BLOCK_COUNT){
        *grid_size = MAX_BLOCK_COUNT;
        load_per_block = need_blocks/MAX_BLOCK_COUNT;
        rem_blocks = need_blocks%MAX_BLOCK_COUNT;
    }
    else{
        *grid_size = need_blocks;
        load_per_block = 1;
        rem_blocks = 0;
    }

    #if MEMORY_MODEL == STD_MEMORY
    workload = (int*)malloc(sizeof(int)*(*grid_size)); 
    if(!workload){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    #endif
    

    for(int i = 0; i<*grid_size; i++){
        if(rem_blocks>i){
            workload[i] = load_per_block+1;
        }
        else{
            workload[i] = load_per_block;
        }
        
    }

    return workload;

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

