/////////////////////////////////////////////////////////////////////////////////////
/* Data Types */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef STDIO
#define STDIO
#include <stdio.h>
#endif


#define UP 1;
#define DOWN 0;



// Structure to represent points with a x and y coordinate
typedef struct Point{
    double x;
    double y;
}Point;


// Structure to represent the line between two points
// y = kx+d and p, q are on the line
typedef struct Line{
    Point p;
    Point q; 
    double k;
    double d;
}Line;


// Point_array contains an array for points and saves the current size and index
typedef struct Point_array{
    Point* array;
    size_t curr_size;
    size_t max_size;
    int index;
}Point_array;


// Hull contains an array for lines and saves the current size and index
typedef struct Hull{
    Line* array;
    size_t curr_size;
    size_t max_size;
    int index;
}Hull;