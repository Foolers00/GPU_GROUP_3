/////////////////////////////////////////////////////////////////////////////////////
/* GENERAL FUNCTIONS */
/////////////////////////////////////////////////////////////////////////////////////

#ifndef STDIO
#define STDIO
#include <stdio.h>
#endif

#ifndef STDLIB
#define STDLIB
#include <stdlib.h>
#endif

#ifndef TIME
#define TIME
#include <time.h>
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
Point max_distance(Line l, Point_array* points);


// returns true if the Point z is above the Line l
// and false if it is below
bool check_above(Line l, Point z);

// generates a Point array with random values from a lower bound(l_bound) to 
// an upper bound(u_bound), init_point_array must be called before
void generate_random_points(Point_array* points, double l_bound, double u_bound);


// sets p(min x) and q(max x) to the two points from the array points that must be on the Hull
void points_on_hull(Point_array* points, Point* p, Point* q);