#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_device_runtime_api.h>


#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <thread>

#define DEVICE 0



enum BalancingVersion
{
	WITHOUT_SORT_ON_CPU,
	WITH_SORT_ON_CPU,
	WITHOUT_SORT_ON_GPU,
	WITH_SORT_ON_GPU
};

struct BalancingInfo 
{
	int numberOfMemoryCopies;
	float time;
	int numThreads;
	int maxNumberOfBoxesPerThread;
	int numAllBoxes;
	int numAverageBoxes;
	BalancingVersion version;
};

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


		
/**
*	Test CUDA kernel
*/
__global__ void testCUDARun();

/**
*	Test time of GPU kernel runs
*/
void testGPUKernelRun(const int numRuns, dim3 gridSize, dim3 blockSize);


BalancingInfo balancingOnCPU(double* boxes, int *workLen,int n, int m, int dim);
BalancingInfo balancingOnCPU2(int n, int m, int dim);
void sortQuickRecursive(int *indexes,int *ar,  const int n);
void quickSortBase(int *indexes,int *ar, const int l, const int r);
BalancingInfo balancingOnCPU_v3(double* boxes, int *workLen, int n, int m, int dim);
BalancingInfo balancingOnCPU_v2(double* boxes, int *workLen, int n, int m, int dim);
void initializeBoxes(double* boxes, int *workLen, int n, int m, int dim);

BalancingInfo balancingOnGPU_v1(double* boxes, int *workLen, int n, int m, int dim);
BalancingInfo balancingOnGPU_v2(double* boxes, int *workLen, int n, int m, int dim);


__global__ void balancingCUDA_v1(double *boxes, const int dim, int *workLen, int *countMemoryCopies, const int m);
__global__ void balancingCUDA_v2(double *boxes, const int dim, int *workLen, int *countMemoryCopies, const int m);

__device__ void quickSortBaseGPU(int *indexes,int *ar, const int l, const int r);
__device__ void sortQuickRecursiveGPU(int *indexes,int *ar,  const int n);
