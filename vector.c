/////////////////////////////////////////////////////////////////////////////////////
/* VECTOR */
/////////////////////////////////////////////////////////////////////////////////////



#ifndef VECTOR
#define VECTOR
#include "vector.h"
#endif


Vector init_vector(Point p, Point q){
    Vector v;

    v._x = q.x-p.x;
    v._y = q.y-p.y;

    return v;
}


double cross_product(Vector v1, Vector v2){
    
    return (v1._x*v2._y)-(v1._y*v2._x);

}


Vector vector_minus(Vector v1, Vector v2){

    return (Vector) {._x = v1._x-v2._x, ._y = v1._y-v2._y};

}


double vector_multiply(Vector v1, Vector v2){
    return ((v1._x*v2._x)+(v1._y*v2._y));
}


Vector vector_scale(Vector v, double scale){

    v._x = v._x*scale;
    v._y = v._y*scale;
    return v;

}


double vector_abs(Vector v){

    return sqrt(pow(v._x, 2)+pow(v._y, 2));

}