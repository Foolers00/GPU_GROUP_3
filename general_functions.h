/////////////////////////////////////////////////////////////////////////////////////
/* GENERAL FUNCTIONS */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef STDIO
#define STDIO
#include <stdio.h>
#endif

#ifndef BOOL
#define BOOL
#include <stdbool.h>
#endif

#ifndef FLOAT
#define FLOAT
#include <float.h>
#endif

#ifndef DATA_TYPES
#define DATA_TYPES
#include "data_types.h"
#endif

// calculates the minimal distance from a Point z to the Line l 
double distance(Line l, Point z);


// returns the Point with maximal distance to the Line l
Point max_distance(Line l, Point* point_array);


// returns true if the Point z is above the Line l
// and false if it is below
bool check_above(Line l, Point z);

// generates a Point array with random values from l_bound to u_bound
// which is size big
Point_array* generate_random_points(int size, double l_bound, double u_bound);


// sets p(min x) and q(max x) to the two points from the array points that must be on the Hull
void points_on_hull(Point_array* points, Point* p, Point* q);