#ifndef INTERVAL_H
#define INTERVAL_H

#include <iostream>
#include <stdio.h>

#define DEVICE 0
#define MAX_NUM_RUNS (1000000)


const int BLOCK_SIZE = 1024;
const int NUM_BLOCKS = 11;
const int SIZE_BUFFER_PER_THREAD = 1024; 
const int TYPE_CUDA_OPTIMIZATION = 2;
const int MAX_GPU_ITER = 1;
const int BORDER_BALANCE = 1;
const int MAX_ITER_BEFORE_BALANCE = 1;




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
