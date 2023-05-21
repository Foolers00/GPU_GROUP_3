/////////////////////////////////////////////////////////////////////////////////////
/* Data Types */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef STDIO
#define STDIO
#include <stdio.h>
#endif


#define ABOVE 1
#define ON 0
#define BELOW -1




/*
    Structure to represent points with a x and y coordinate
*/
typedef struct Point{
    double x;
    double y;
}Point;


/*
    Structure to represent the line between two points
    p, q are on the line and v shows the orientation
*/
typedef struct Line{
    Point p;
    Point q;
}Line;

/*
    Structure to represent the Vector between two points
*/
typedef struct Vector{
    double _x;
    double _y;
}Vector;


/*
    Point_array contains an array for points and saves the current size and index
*/
typedef struct Point_array{
    Point* array;
    size_t curr_size;
    size_t max_size;
}Point_array;


/*
    Hull contains an array for lines and saves the current size and index
*/
typedef struct Hull{
    Line* array;
    size_t curr_size;
    size_t max_size;
}Hull;