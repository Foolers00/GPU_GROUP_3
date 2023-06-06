/////////////////////////////////////////////////////////////////////////////////////
/* VECTOR */
/////////////////////////////////////////////////////////////////////////////////////

#define VECTOR

#ifndef DATA_TYPES
#define DATA_TYPES
#include "data_types.h"
#endif


#ifndef MATH
#define MATH
#include <math.h>
#endif


/* 
    initializes vector with points p and q by subtracting
    q-p
*/
Vector init_vector(Point p, Point q);


/* 
    calculates cross product between line l and point z
*/
double cross_product(Vector v1, Vector v2);


/*
    calculates subtraction between vector v1 and v2
*/
Vector vector_minus(Vector v1, Vector v2);


/*
    calculates the scale of vector v by scale
*/
Vector vector_scale(Vector v, double scale);


/*
    calculates the absolute value of vector v
*/
double vector_abs(Vector v);


/*
    calculates the multiplication of v1 and v2
*/
double vector_multiply(Vector v1, Vector v2);