#ifndef INTERVAL_H
#define INTERVAL_H

#include <iostream>
#include <stdio.h>

#define DEVICE 2
#define MAX_NUM_RUNS (100)


int const BLOCK_SIZE = 1024;




#define CHECKED_CALL(func)                                     \
    do {                                                       \
        cudaError_t err = (func);                              \
        if (err != cudaSuccess) {                              \
            printf("%s(%d): ERROR: %s returned %s (err#%d)\n", \
                   __FILE__, __LINE__, #func,                  \
                   cudaGetErrorString(err), err);              \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#endif