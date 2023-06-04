/////////////////////////////////////////////////////////////////////////////////////
/* GENERAL FUNCTIONS PARALLEL */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef GENERAL_FUNCTIONS_PAR
#define GENERAL_FUNCTIONS_PAR
#include "general_functions_par.h"
#endif






Hull* quickhull_split_par(Point_array* points, Line l, int side){


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

    workload = (int*)malloc(sizeof(int)*(*grid_size)); 
    if(!workload){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    

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


