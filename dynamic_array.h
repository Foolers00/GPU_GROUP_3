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

// Adds a point z to the point array, resizes array if half is full
void add_to_point_array(Point_array* points,  Point z);

// Adds line l to the hull, resizes array if half is full
void add_to_hull(Hull* hull, Line l);

// combines two hulls by creating new dynamic array
Hull* combine_hull(Hull* hull_1, Hull* hull_2);

