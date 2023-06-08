/////////////////////////////////////////////////////////////////////////////////////
/* Data Types Parallel */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef DATA_TYPES_PAR
#define DATA_TYPES_PAR
#include "data_types_par.h"
#endif




Point_array_par* init_point_array_par(size_t size){
    
    Point_array_par* points;

    #if MEMORY_MODEL == STD_MEMORY
    points = (Point_array_par*)malloc(sizeof(Point_array_par));
    if(!points){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    #elif MEMORY_MODEL == PINNED_MEMORY
    CHECK(cudaMallocHost((Point_array_par**)&points, sizeof(Point_array_par)));
    #elif MEMORY_MODEL == ZERO_MEMORY
    CHECK(cudaHostAlloc((Point_array_par**)&points, sizeof(Point_array_par), cudaHostAllocMapped));
    #endif

    points->size = size;
    points->array = NULL;

    if(size == 0){
        return points;
    }

    #if MEMORY_MODEL == STD_MEMORY
    points->array = (Point*)malloc(points->size*sizeof(Point));
    if(!points->array){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    #elif MEMORY_MODEL == PINNED_MEMORY
    CHECK(cudaMallocHost((Point**)&points->array, points->size*sizeof(Point)));
    #elif MEMORY_MODEL == ZERO_MEMORY
    CHECK(cudaHostAlloc((Point**)&points->array, points->size*sizeof(Point), cudaHostAllocMapped));
    #endif

    return points;

}


Point_array_par* init_point_array_par_gpu(size_t size){
    
    Point_array_par* points;

    #if MEMORY_MODEL == STD_MEMORY
    points = (Point_array_par*)malloc(sizeof(Point_array_par));
    if(!points){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    #elif MEMORY_MODEL == PINNED_MEMORY
    CHECK(cudaMallocHost((Point_array_par**)&points, sizeof(Point_array_par)));
    #elif MEMORY_MODEL == ZERO_MEMORY
    CHECK(cudaHostAlloc((Point_array_par**)&points, sizeof(Point_array_par), cudaHostAllocMapped));
    #endif

    points->size = size;
    points->array = NULL;

    if(size == 0){
        return points;
    }

    CHECK(cudaMalloc((Line **)&points->array, points->size*sizeof(Point)));

    return points;

}


void free_point_array_par(Point_array_par* points){
    if(!points){
        return;
    }

    #if MEMORY_MODEL == STD_MEMORY
    if(points->array){
        free(points->array);
    }
    free(points);
    #elif MEMORY_MODEL == PINNED_MEMORY || MEMORY_MODEL == ZERO_MEMORY
    if(points->array){
        CHECK(cudaFreeHost(points->array));
    }
    CHECK(cudaFreeHost(points));
    #endif
}


void free_point_array_par_gpu(Point_array_par* points){
    if(!points){
        return;
    }

    #if MEMORY_MODEL == STD_MEMORY
    if(points->array){
        CHECK(cudaFree(points->array));
    }
    free(points);
    #elif MEMORY_MODEL == PINNED_MEMORY || MEMORY_MODEL == ZERO_MEMORY
    if(points->array){
        CHECK(cudaFree(points->array));
    }
    CHECK(cudaFreeHost(points));
    #endif
}



Hull_par* init_hull_par(size_t size){
    
    Hull_par* hull = NULL;
    
    #if MEMORY_MODEL == STD_MEMORY
    hull = (Hull_par*)malloc(sizeof(Hull_par));
    if(!hull){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    #elif MEMORY_MODEL == PINNED_MEMORY
    CHECK(cudaMallocHost((Hull_par**)&hull, sizeof(Hull_par)));
    #elif MEMORY_MODEL == ZERO_MEMORY
    CHECK(cudaHostAlloc((Hull_par**)&hull, sizeof(Hull_par), cudaHostAllocMapped));
    #endif

    hull->size = size;
    hull->array = NULL;

    if(size == 0){
        return hull;
    }
    
    #if MEMORY_MODEL == STD_MEMORY
    hull->array = (Line*)malloc(hull->size*sizeof(Line));
    if(!hull->array){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    #elif MEMORY_MODEL == PINNED_MEMORY
    CHECK(cudaMallocHost((Hull_par**)&hull->array, hull->size*sizeof(Line)));
    #elif MEMORY_MODEL == ZERO_MEMORY
    CHECK(cudaHostAlloc((Hull_par**)&hull->array, hull->size*sizeof(Line), cudaHostAllocMapped));
    #endif

    return hull;

}


Hull_par* init_hull_par_gpu(size_t size){
    
    Hull_par* hull = NULL;
    
    #if MEMORY_MODEL == STD_MEMORY
    hull = (Hull_par*)malloc(sizeof(Hull_par));
    if(!hull){
        fprintf(stderr, "Malloc failed");
        exit(1);
    }
    #elif MEMORY_MODEL == PINNED_MEMORY
    CHECK(cudaMallocHost((Hull_par**)&hull, sizeof(Hull_par)));
    #elif MEMORY_MODEL == ZERO_MEMORY
    CHECK(cudaHostAlloc((Hull_par**)&hull, sizeof(Hull_par), cudaHostAllocMapped));
    #endif

    hull->size = size;
    hull->array = NULL;

    CHECK(cudaMalloc((Line **)&hull->array, hull->size*sizeof(Line)));

    return hull;

}





void free_hull_par(Hull_par* hull){
    if(!hull){
        return;
    }

    #if MEMORY_MODEL == STD_MEMORY
    if(hull->array){
        free(hull->array);
    }
    free(hull);
    #elif MEMORY_MODEL == PINNED_MEMORY || MEMORY_MODEL == ZERO_MEMORY
    if(hull->array){
        CHECK(cudaFreeHost(hull->array));
    }
    CHECK(cudaFreeHost(hull));
    #endif
}



void free_hull_par_gpu(Hull_par* hull){
    if(!hull){
        return;
    }

    #if MEMORY_MODEL == STD_MEMORY
    if(hull->array){
        CHECK(cudaFree(hull->array));
    }
    free(hull);
    #elif MEMORY_MODEL == PINNED_MEMORY || MEMORY_MODEL == ZERO_MEMORY
    if(hull->array){
        CHECK(cudaFree(hull->array));
    }
    CHECK(cudaFreeHost(hull));
    #endif
}