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
#include <cstring>

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
*	Test CUDA kernel for GPU kernel runs
*	@param boxes the test boxes
*/
__global__ void testCUDARun(double *boxes);

/**
*	Test time of GPU kernel runs
*	@param numRuns the number of cuda testing calls
*	@param gridSize CUDA grid's size
*	@param blockSize CUDA block's size
*/
void testGPUKernelRun(const int numRuns, dim3 gridSize, dim3 blockSize);

/**
*	Test Transfer data to device
*	@param numRuns the number of cuda testing calls
*	@param gridSize CUDA grid's size
*	@param blockSize CUDA block's size
*	@param dataVolume the volume of data for transering to CUDA
*	@param fileName file to save data
*	@param isToFile if we should save data to file
*/
void testGPUTransferDataToDevice(const int numRuns, dim3 gridSize, dim3 blockSize, long long dataVolume, char* fileName, bool isToFile);

/**
*	Test transfer data from device
*	@param numRuns the number of cuda testing calls
*	@param gridSize CUDA grid's size
*	@param blockSize CUDA block's size
*	@param dataVolume the volume of data for transering to CUDA
*	@param fileName file to save data
*	@param isToFile if we should save data to file
*/
void testGPUTransferDataFromDevice(const int numRuns, dim3 gridSize, dim3 blockSize, long long dataVolume, char* fileName, bool isToFile);

/**
*	Test GPU inner memory access
*	@param numRuns the number of cuda testing calls
*	@param gridSize CUDA grid's size
*	@param blockSize CUDA block's size
*	@param fileName file to save data
*	@param isToFile if we should save data to file
*	@param partSize the number of values to copy by one thread
*/
void testGPUMemoryAccess(const int numRuns, dim3 gridSize, dim3 blockSize, char* fileName, bool isToFile, int partSize);

/**
*	Test CUDA kernel for GPU multiple thread memory access withmemcpy
*	@param ar1 the test array copy from
*	@param ar2 the test array copy To
*	@param partSize the number of values to copy by one thread
*/
__global__ void testCUDAMemoryAccessRunMultiThread_v1(double *ar1, double *ar2, int partSize);

/**
*	Test CUDA kernel for GPU multiple thread memory access without memcpy
*	@param ar1 the test array copy from
*	@param ar2 the test array copy To
*	@param partSize the number of values to copy by one thread
*/
__global__ void testCUDAMemoryAccessRunMultiThread_v2(double *ar1, double *ar2, int partSize);


/**
*	Test CUDA kernel for GPU single thread memory access with memcpy
*	@param ar1 the test array copy from
*	@param ar2 the test array copy To
*	@param partSize the number of values to copy by one thread
*/
__global__ void testCUDAMemoryAccessRunSingleThread_v1(double *ar1, double *ar2, int partSize);

/**
*	Test CUDA kernel for GPU single thread memory access without memcpy
*	@param ar1 the test array copy from
*	@param ar2 the test array copy To
*	@param partSize the number of values to copy by one thread
*/
__global__ void testCUDAMemoryAccessRunSingleThread_v2(double *ar1, double *ar2, int partSize);

/**
*	Initialize test boxes for balancing
*	@param boxes array for test boxes
*	@param workLen array for numbers of boxes per thread
*	@param n the number of threads
*	@param m max number of boxes per thread
*	@param dim max number of boxes per thread
*/
void initializeBoxes(double* boxes, int *workLen, int n, int m, int dim);

/**
*	Quick sort algorithm for balancing
*	@param indexes the array of boxes' indexes before sorting
*	@param ar the array of work boxes numbers
*	@param l left index of array
*	@param r right index of array
*/
void quickSortBase(int *indexes,int *ar, const int l, const int r);

/**
*	Quick sort algorithm for balancing
*	@param indexes the array of boxes' indexes before sorting
*	@param ar the array of work boxes numbers
*	@param n the number of elements in array
*/
void sortQuickRecursive(int *indexes,int *ar,  const int n);

/**
*	Quick sort algorithm for balancing on GPU
*	@param indexes the array of boxes' indexes before sorting
*	@param ar the array of work boxes numbers
*	@param l left index of array
*	@param r right index of array
*/
__device__ void quickSortBaseGPU(int *indexes,int *ar, const int l, const int r);

/**
*	Quick sort algorithm for balancing on GPU
*	@param indexes the array of boxes' indexes before sorting
*	@param ar the array of work boxes numbers
*	@param n the number of elements in array
*/
__device__ void sortQuickRecursiveGPU(int *indexes,int *ar,  const int n);





BalancingInfo balancingOnCPU(double* boxes, int *workLen,int n, int m, int dim);
BalancingInfo balancingOnCPU2(int n, int m, int dim);


BalancingInfo balancingOnCPU_v1(double* boxes, int *workLen, int n, int m, int dim);
BalancingInfo balancingOnCPU_v2(double* boxes, int *workLen, int n, int m, int dim);


BalancingInfo balancingOnGPU_v1(double* boxes, int *workLen, int n, int m, int dim);
BalancingInfo balancingOnGPU_v2(double* boxes, int *workLen, int n, int m, int dim);


__global__ void balancingCUDA_v1(double *boxes, const int dim, int *workLen, int *countMemoryCopies, const int m);
__global__ void balancingCUDA_v2(double *boxes, const int dim, int *workLen, int *countMemoryCopies, const int m);

__device__ void quickSortBaseGPU(int *indexes,int *ar, const int l, const int r);
__device__ void sortQuickRecursiveGPU(int *indexes,int *ar,  const int n);
