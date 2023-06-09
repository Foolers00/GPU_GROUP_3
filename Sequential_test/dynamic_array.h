/////////////////////////////////////////////////////////////////////////////////////
/* DYNAMIC ARRAY */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef STDIO
#define STDIO
#include <stdio.h>
#endif

#ifndef STDLIB
#define STDLIB
#include <stdlib.h>
#endif

#ifndef STRING
#define STRING
#include <string.h>
#endif

#ifndef DATA_TYPES
#define DATA_TYPES
#include "../Data_Types/data_types.h"
#endif

/*
    Adds a point z to the point array, resizes array if half is full
*/
void add_to_point_array(Point_array* points,  Point z);

/*
    Adds line l to the hull, resizes array if half is full
*/
void add_to_hull(Hull* hull, Line l);

/*
    combines two hulls by creating new dynamic array
*/
Hull* combine_hull(Hull* hull_1, Hull* hull_2);

/*
    frees the array inside points and then points itself
*/
void free_point_array(Point_array* points);

/*
    frees the array inside hull and then hull itself
*/
void free_hull(Hull* hull);


/*
    prints the values in the Point_array points
*/
void print_point_array(Point_array* points);