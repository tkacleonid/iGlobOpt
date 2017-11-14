#ifndef INTERVAL_H
#define INTERVAL_H

#include <iostream>
#include <stdio.h>

#define DEVICE 3
#define MAX_NUM_RUNS (1000000)


const int BLOCK_SIZE = 1024;
const int NUM_BLOCKS = 1;
const int SIZE_BUFFER_PER_THREAD = 1024; 
const int TYPE_CUDA_OPTIMIZATION = 1;
const int MAX_GPU_ITER = 100000;
const int BORDER_BALANCE = 2;




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