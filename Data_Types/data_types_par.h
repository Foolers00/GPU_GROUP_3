/////////////////////////////////////////////////////////////////////////////////////
/* Data Types */
/////////////////////////////////////////////////////////////////////////////////////

#define DATA_TYPES_PAR

#ifndef STDIO
#define STDIO
#include <stdio.h>
#endif

#ifndef DATA_TYPES
#define DATA_TYPES
#include "data_types.h"
#endif


#define MAX_BLOCK_COUNT 65535
#define BLOCKSIZE 512
#define MAX_BLOCK_COUNT_SHIFT MAX_BLOCK_COUNT*2*BLOCKSIZE
#define INCLUSIVE 1
#define EXCLUSIVE 0


#define STD_MEMORY 1
#define PIN_MEMORY 2
#define ZERO_MEMORY 3

#define MEMORY_MODEL STD_MEMORY



#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(1);                                               \
        }                                                          \
    }





/*
    Point_array contains an array for points and saves the current size and index
*/
typedef struct Point_array_par{
    Point* array;
    size_t size;
}Point_array_par;


/*
    Hull contains an array for lines and saves the current size and index
*/
typedef struct Hull_par{
    Line* array;
    size_t size;
}Hull_par;