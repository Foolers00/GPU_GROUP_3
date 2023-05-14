/////////////////////////////////////////////////////////////////////////////////////
/* MAIN */
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

#ifndef GENERAL_FUNCTIONS
#define GENERAL_FUNCTIONS
#include "general_functions.h"
#endif


#ifndef DYNAMIC_ARRAY
#define DYNAMIC_ARRAY
#include "dynamic_array.h"
#endif

#ifndef TEST
#define TEST
#include "test.h"
#endif



int main(){

    Point x1,x2,x3,x4;
    Line l;
    l.d = 2.0;
    l.k = 0.0;
    x1 = (Point) {1.0, 3.0};
    x2 = (Point) {4.0, 0.0};
    x3 = (Point) {2.0,2.0};
    x4 = (Point) {2.0, 5.0};

    Point_array points = (Point_array){(Point*) malloc(sizeof(Point) * 4), 4, 4};
    points.array[0] = x1;
    points.array[1] = x2;
    points.array[2] = x3;
    points.array[3] = x4;

    for (int i = 0; i < 4; i++) {
        printf("Location of x_%i: %s\n", i,
               check_point_location(l, points.array[i]) == ABOVE ? "ABOVE" : check_point_location(l, points.array[i]) == ON ? "ON"
                                                                                                              : "BELOW");
        printf("Distance of x_%i to l: %f\n", i, distance(l, points.array[i]));
    }
    printf("point of maximal distance: (%f, %f)\n", max_distance(l, &points).x, max_distance(l, &points).y);
    return 0;
}