/////////////////////////////////////////////////////////////////////////////////////
/* Data Types */
/////////////////////////////////////////////////////////////////////////////////////

#define DATA_TYPES

#ifndef STDIO
#define STDIO
#include <stdio.h>
#endif

#ifndef STDLIB
#define STDLIB
#include <stdlib.h>
#endif

#ifndef BOOL
#define BOOL
#include <stdbool.h>
#endif


#define ABOVE 1
#define ON 0
#define BELOW -1

#define OP_MIN 0
#define OP_MAX 1




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


/*
    Returns malloced point array and initializes the values size for curr_size, 
    max_size/2 for the threshold to expand and mallocs array inside point array 
    with max_size size 
*/
Point_array* init_point_array(size_t max_size);

/* 
    Returns malloced Hull and initializes the values size for curr_size, max_size/2 
    for the threshold to expand and mallocs line array curr_size big 
*/
Hull* init_hull(size_t max_size);



/*
    returns true if lines are same otherwise false
*/
bool compare_lines(Line l_1, Line l_2);


/*
    returns false if lines are same otherwise false
*/
bool compare_points(Point p_1, Point p_2);