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
#include "data_types.h"
#endif

// Initializes the values size for curr_size, max_size for the threshold to expand 
// and mallocs point array urr_size big 
void init_point_array(Point_array* points, size_t curr_size, size_t max_size);

// Initializes the values size for curr_size, max_size for the threshold to expand
// and mallocs line array curr_size big 
void init_hull(Hull* hull, size_t curr_size, size_t max_size);

// Adds a point z to the point array, resizes array if half is full
void add_to_point_array(Point_array* points,  Point z);

// Adds line l to the hull, resizes array if half is full
void add_to_hull(Hull* hull, Line l);

// combines two hulls by creating new dynamic array
Hull* combine_hull(Hull* hull_1, Hull* hull_2);