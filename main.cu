/////////////////////////////////////////////////////////////////////////////////////
/* MAIN */
/////////////////////////////////////////////////////////////////////////////////////


#ifndef TEST
#define TEST
#include "Test/test.h"
#endif



int main(){

    // Point_array* points = test_sequence_4_1();// generate_random_points(8, -200, 800);
    // Hull* hull = quickhull(points);

    //test_sequence_2();
    //test_prescan();
    //test_prescan_gpu();

    //test_minmax_cuda();

    //validate_minmax();

    test_split();

    return 0;
}
