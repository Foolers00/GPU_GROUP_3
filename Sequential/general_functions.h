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
#include "../Data_Types/data_types.h"
#endif

#ifndef DYNAMIC_ARRAY
#define DYNAMIC_ARRAY
#include "dynamic_array.h"
#endif

#ifndef MATH
#define MATH
#include <math.h>
#endif

#ifndef VECTOR
#define VECTOR
#include "../Data_Types/vector.h"
#endif


#ifndef OMP
#define OMP
#include <omp.h>
#endif


/*
    calculates the hull for the poitns in Point_array
*/
Hull* quickhull(Point_array* points);

/*
    calculates hull for all points in Point_array that are above/below
    the Line l depending on the value of side (UP == 1, DOWN == 0)
*/
Hull* quickhull_split(Point_array* points, Line l, int side);

/*
    calculates line which is made up of points p and q
*/
Line init_line(Point p, Point q);


/*
    calculates the minimal distance from a Point z to the Line l 
*/
double distance(Line l, Point z);


/*
    returns the Point with maximal distance to the Line l
*/
Point max_distance(Line l, Point_array* points);


/*
    returns ABOVE if the Point z is above the Line l,
    ON if it is one the Line l and BELOW if is is below
    the Line l
*/
int check_point_location(Line l, Point z);



/*
    generates a Point array with random values from a lower bound(l_bound) to 
    an upper bound(u_bound), no duplicate points
*/
Point_array* generate_random_points(size_t num_of_points, double l_bound, double u_bound);

Point_array* generate_random_points_on_circle(size_t num_of_points, double radius);

bool find_point_array(Point_array* points, Point p);


/*
    sets p(min x) and q(max x) to the two points from the array points that must be on the Hull
*/
void points_on_hull(Point_array* points, Point* p, Point* q);


