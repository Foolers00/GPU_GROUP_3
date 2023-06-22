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

    //test_max_distance_cuda();

    //validate_minmax();

    //test_split();

    //test_combinehull();

    // test_quickhull();

    //test_thrust_quickhull();
    
    // test_quickhull_performance();

    test_memory_model();

    // Print the value of MEMORY_MODEL
    #ifdef MEMORY_MODEL
    printf("MEMORY_MODEL: %d\n", MEMORY_MODEL);
    #else
    printf("MEMORY_MODEL is not defined\n");
    #endif

    //#ifdef SET_MEMORY_MODEL
    //printf("SET_MEMORY_MODEL: %d\n", SET_MEMORY_MODEL);
    //#else
    //printf("SET_MEMORY_MODEL is not defined\n");
    //#endif

    // Print the value of BLOCKSIZE
    std::cout << "BLOCKSIZE: " << BLOCKSIZE << std::endl;

    return 0;
}