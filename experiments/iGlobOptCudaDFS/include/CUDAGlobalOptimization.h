#ifndef __CUDAGLOBALOPTIMIZATION_H__
#define __CUDAGLOBALOPTIMIZATION_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_device_runtime_api.h>


#include "interval.h"
#include "CUDAFunctionIntervalEstimation.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

/**
*	Global optimization with CUDA kernel version 1 (no balancing)
*	@param inbox pointer to Box
*	@param inDim number of variables
*	@param inEps accuracy
*	@param workLen the array of numbers of boxes per thread
*	@param min the array of fun records per thread
*	@param inRec initial value of function record
*	@param workCount the array of dropped boxes
*/
__global__ void globOptCUDA_v1(double *inBox, int inDim, int *workLen, double *min, double inRec, double inEps, long long *workCounts);


/**
*	Global optimization with CUDA version 1 (one call CUDA kernel and no balancing)
*	@param inbox pointer to Box
*	@param inDim number of variables
*	@param inEps accuracy
*	@param outBox the optimal box (not uses)
*	@param outMin the global minimum
*	@param status the status of finishing (not uses)
*	@param funRecord initial value of function record
*	@param filename the file name to write results
*/
void fnGetOptValueWithCUDA_v1(double *inBox, const int inDim, const double inEps, double *outBox, double*outMin, int *status, double funRecord, char* filename)
{
	int numBoxes = BLOCK_SIZE*NUM_BLOCKS;
	double *boxes =  new double[numBoxes*(inDim*2+3)*SIZE_BUFFER_PER_THREAD];
	double h;
	int hInd;
	int *workLen;
	double *mins;
	int i,n;	
	double funcMin = funRecord;

	*status = 1;

	workLen = new int[numBoxes];
	mins = new double[numBoxes];

	h = inBox[1] - inBox[0];
	hInd = 0;
	
	std::ofstream outfile;
	outfile.open(filename,std::ios_base::app);
	if(outfile.fail())
		throw std::ios_base::failure(std::strerror(errno));
		
	for (i = 0; i < inDim; i++) {
		if(h < inBox[i*inDim + 1] - inBox[i*inDim]) {
			h = inBox[i*inDim + 1] - inBox[i*inDim];
			hInd = i;
		}
	}

	for (n = 0; n < numBoxes; n++) {
		for(i = 0; i < inDim; i++) {
			if (i == hInd) {
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2] = inBox[i*2] + h/numBoxes*n;
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2 + 1] = inBox[i*2] + h/numBoxes*(n+1);
			}
			else {
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2] = inBox[i*2];
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2 + 1] = inBox[i*2 + 1];
			}
		}
	}
	
	double *dev_inBox = 0;
	int *dev_workLen = 0;
	long long *dev_workCounts = 0;
	double *dev_mins = 0;

	int GridSize = NUM_BLOCKS;
	int numThreads = numBoxes;
	int sizeInBox = numThreads*(inDim*2+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD;
	
	float time, timeAll;
	long long *workCounts = new long long[numThreads*sizeof(int)];
	
	for (i = 0; i < numThreads; i++) {
		workLen[i] = 1;
		workCounts[i] = 0;
	}

	cudaEvent_t start, stop;
	
	CHECKED_CALL(cudaSetDevice(DEVICE));
	CHECKED_CALL(cudaDeviceReset());	
    CHECKED_CALL(cudaMalloc((void **)&dev_inBox, sizeInBox));
	CHECKED_CALL(cudaMalloc((void **)&dev_workLen, numThreads*sizeof(int)));
	CHECKED_CALL(cudaMalloc((void **)&dev_mins, numThreads*sizeof(double)));
	CHECKED_CALL(cudaMalloc((void **)&dev_workCounts, numThreads*sizeof(long long)));
	
	timeAll = 0;
	long long wc = 0;
	long long ls = 0;

	auto startCPU = std::chrono::high_resolution_clock::now();

	CHECKED_CALL(cudaEventCreate(&start));
	CHECKED_CALL(cudaEventCreate(&stop));
	CHECKED_CALL(cudaMemcpy(dev_inBox, boxes, numBoxes*(2*inDim+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD, cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_workLen, workLen, numThreads*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_workCounts, workCounts, numThreads*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaEventRecord(start, 0));
		
	globOptCUDA_v1<<<GridSize, BLOCK_SIZE>>>(dev_inBox,inDim,dev_workLen,dev_mins,funcMin,inEps,dev_workCounts);

	CHECKED_CALL(cudaGetLastError());
	CHECKED_CALL(cudaEventRecord(stop, 0));
	CHECKED_CALL(cudaEventSynchronize(stop));
	CHECKED_CALL(cudaDeviceSynchronize());

	CHECKED_CALL(cudaMemcpy(boxes, dev_inBox, numBoxes*(2*inDim+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD, cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(workLen, dev_workLen, numThreads*sizeof(int), cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(mins, dev_mins, numThreads*sizeof(double), cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(workCounts, dev_workCounts, numThreads*sizeof(long long), cudaMemcpyDeviceToHost));

	CHECKED_CALL(cudaEventElapsedTime(&time, start, stop)); 
		
	ls = 0;	
	for (int j = 0; j < numThreads; j++) {
		wc+=workCounts[j];
		ls += workLen[j];
		workCounts[j] = 0;
	}

	funcMin = mins[0];		
	for (int j  = 1; j < numThreads; j++) {
		if(funcMin > mins[j]) funcMin = mins[j];
	}
	CHECKED_CALL(cudaEventDestroy(start));
	CHECKED_CALL(cudaEventDestroy(stop));
	
	timeAll += time;
	
	auto endCPU = std::chrono::high_resolution_clock::now();

	CHECKED_CALL(cudaFree(dev_inBox));
    CHECKED_CALL(cudaFree(dev_workLen));
	CHECKED_CALL(cudaFree(dev_mins));
	CHECKED_CALL(cudaFree(dev_workCounts));
	
	std::cout << "EPS = " << inEps << "\n";
	std::cout << "MIN = " << funcMin << "\n";
	std::cout << "wc = " << wc << "\n";
	std::cout <<  "timeAll = " << timeAll << "\n";
	std::cout <<  "timeAllCPU = " << (std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU)).count() << "\n";
	
	outfile << inDim << "\t" << inEps << "\t" << timeAll << "\t" << (std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU)).count() << "\t" << wc << "\n";
	outfile.close();

}

/**
*	Global optimization with CUDA kernel version 1 (no balancing)
*	@param inbox pointer to Box
*	@param inDim number of variables
*	@param inEps accuracy
*	@param workLen the array of numbers of boxes per thread
*	@param min the array of fun records per thread
*	@param inRec initial value of function record
*	@param workCount the array of dropped boxes
*/
__global__ void globOptCUDA_v1(double *inBox,  int inDim, int *workLen, double *min, double inRec, double inEps, long long *workCounts)

{
	__shared__ double min_s[BLOCK_SIZE];
	__shared__ int workLen_s[BLOCK_SIZE];
	__shared__ int count[BLOCK_SIZE];
	
	int i,bInd, hInd, n;
	double  h;
	
	int threadId = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	
	workLen_s[threadIdx.x] = workLen[threadId];
	min_s[threadIdx.x] = inRec;	
	count[threadIdx.x] = 0;
	

	__syncthreads();

	n = 0;
	while (workLen_s[threadIdx.x] > 0 && workLen_s[threadIdx.x] < SIZE_BUFFER_PER_THREAD && count[threadIdx.x] < MAX_GPU_ITER) {
		bInd = threadId*SIZE_BUFFER_PER_THREAD*(2*inDim+3) + (workLen_s[threadIdx.x] - 1)*(2*inDim+3);
		fnCalcFunLimitsStyblinski_CUDA(inBox + bInd, inDim);	
		if (min_s[threadIdx.x] > inBox[bInd + 2*inDim + 2]) {
			min_s[threadIdx.x] = inBox[bInd + 2*inDim + 2];
		}
			
		if (min_s[threadIdx.x] - inBox[bInd + 2*inDim] < inEps) {
			--workLen_s[threadIdx.x];
			n++;
		}
		else {	
			hInd = 0;
			h = inBox[bInd + 1] - inBox[bInd];
			for (i = 0; i < inDim; i++) {
				if ( h < inBox[bInd + i*2 + 1] - inBox[bInd + i*2]) {
					h = inBox[bInd + i*2 + 1] - inBox[bInd + i*2];
					hInd = i;
				}
			}
			for (i = 0; i < inDim; i++) {
				if(i == hInd) {
					inBox[bInd + i*2 + 1] = inBox[bInd + i*2] + h/2.0;
					inBox[bInd + 2*inDim + 3 + i*2] = inBox[bInd + i*2] + h/2.0;
					inBox[bInd + 2*inDim + 3 + i*2 + 1] = inBox[bInd + i*2] + h;
				}
				else {
					inBox[bInd + 2*inDim + 3 + i*2] = inBox[bInd + i*2];
					inBox[bInd + 2*inDim + 3 + i*2 + 1] = inBox[bInd + i*2 + 1];
				}
			}
			++workLen_s[threadIdx.x];
		}		
		++count[threadIdx.x];	
	}
	
	workLen[threadId] = workLen_s[threadIdx.x];
	min[threadId] = min_s[threadIdx.x];
	workCounts[threadId] = n;	
}

/**
*	Global optimization with CUDA version 2 (with only CPU balancing)
*	@param inbox pointer to Box
*	@param inDim number of variables
*	@param inEps accuracy
*	@param outBox the optimal box (not uses)
*	@param outMin the global minimum
*	@param status the status of finishing (not uses)
*	@param funRecord initial value of function record
*	@param filename the file name to write results
*/
void fnGetOptValueWithCUDA_v2(double *inBox, const int inDim, const double inEps, double *outBox, double*outMin, int *status, double funRecord, char* filename)
{
	int numBoxes = BLOCK_SIZE*NUM_BLOCKS;
	double *boxes =  new double[numBoxes*(inDim*2+3)*SIZE_BUFFER_PER_THREAD];
	double h;
	int hInd;
	int *workLen;
	double *mins;
	int i,n;	
	double funcMin = funRecord;

	*status = 1;

	workLen = new int[numBoxes];
	mins = new double[numBoxes];

	h = inBox[1] - inBox[0];
	hInd = 0;
	
	std::ofstream outfile;
	outfile.open(filename,std::ios_base::app);
	if(outfile.fail())
		throw std::ios_base::failure(std::strerror(errno));
		
	for (i = 0; i < inDim; i++) {
		if(h < inBox[i*inDim + 1] - inBox[i*inDim]) {
			h = inBox[i*inDim + 1] - inBox[i*inDim];
			hInd = i;
		}
	}

	for (n = 0; n < numBoxes; n++) {
		for(i = 0; i < inDim; i++) {
			if (i == hInd) {
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2] = inBox[i*2] + h/numBoxes*n;
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2 + 1] = inBox[i*2] + h/numBoxes*(n+1);
			}
			else {
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2] = inBox[i*2];
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2 + 1] = inBox[i*2 + 1];
			}
		}
	}
	
	double *dev_inBox = 0;
	int *dev_workLen = 0;
	long long *dev_workCounts = 0;
	double *dev_mins = 0;

	int GridSize = NUM_BLOCKS;
	int numThreads = numBoxes;
	int sizeInBox = numThreads*(inDim*2+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD;
	
	float time, timeAll;
	long long *workCounts = new long long[numThreads*sizeof(int)];
	
	for (i = 0; i < numThreads; i++) {
		workLen[i] = 1;
		workCounts[i] = 0;
	}

	cudaEvent_t start, stop;
	
	CHECKED_CALL(cudaSetDevice(DEVICE));
	CHECKED_CALL(cudaDeviceReset());	
    CHECKED_CALL(cudaMalloc((void **)&dev_inBox, sizeInBox));
	CHECKED_CALL(cudaMalloc((void **)&dev_workLen, numThreads*sizeof(int)));
	CHECKED_CALL(cudaMalloc((void **)&dev_mins, numThreads*sizeof(double)));
	CHECKED_CALL(cudaMalloc((void **)&dev_workCounts, numThreads*sizeof(long long)));
	
	timeAll = 0;
	long long wc = 0;
	long long ls = 0;

	auto startCPU = std::chrono::high_resolution_clock::now();

	CHECKED_CALL(cudaEventCreate(&start));
	CHECKED_CALL(cudaEventCreate(&stop));
	
	for (i = 0; i < MAX_NUM_RUNS ; i++) {
		CHECKED_CALL(cudaMemcpy(dev_inBox, boxes, numBoxes*(2*inDim+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD, cudaMemcpyHostToDevice));
		CHECKED_CALL(cudaMemcpy(dev_workLen, workLen, numThreads*sizeof(int), cudaMemcpyHostToDevice));
		CHECKED_CALL(cudaMemcpy(dev_workCounts, workCounts, numThreads*sizeof(int), cudaMemcpyHostToDevice));
		CHECKED_CALL(cudaEventRecord(start, 0));
		
		globOptCUDA_v1<<<GridSize, BLOCK_SIZE>>>(dev_inBox,inDim,dev_workLen,dev_mins,funcMin,inEps,dev_workCounts);

		CHECKED_CALL(cudaGetLastError());
		CHECKED_CALL(cudaEventRecord(stop, 0));
		CHECKED_CALL(cudaEventSynchronize(stop));
		CHECKED_CALL(cudaDeviceSynchronize());

		CHECKED_CALL(cudaMemcpy(boxes, dev_inBox, numBoxes*(2*inDim+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD, cudaMemcpyDeviceToHost));
		CHECKED_CALL(cudaMemcpy(workLen, dev_workLen, numThreads*sizeof(int), cudaMemcpyDeviceToHost));
		CHECKED_CALL(cudaMemcpy(mins, dev_mins, numThreads*sizeof(double), cudaMemcpyDeviceToHost));
		CHECKED_CALL(cudaMemcpy(workCounts, dev_workCounts, numThreads*sizeof(long long), cudaMemcpyDeviceToHost));

		CHECKED_CALL(cudaEventElapsedTime(&time, start, stop)); 
		
		ls = 0;	
		for (int j = 0; j < numThreads; j++) {
			wc+=workCounts[j];
			ls += workLen[j];
			workCounts[j] = 0;
		}

		funcMin = mins[0];		
		for (int j  = 1; j < numThreads; j++) {
			if(funcMin > mins[j]) funcMin = mins[j];
		}
	
		timeAll += time;
		if(ls ==0) break;
		
		int numWorkBoxes = 0;
		int averageBoxesPerThread = 0;
		int curThreadWeTakeBoxesIndex = -1;
		int numBoxesWeTake = 0;

		for (int m = 0; m < numThreads; m++) {
			numWorkBoxes += workLen[m]; 	
		}
		averageBoxesPerThread = numWorkBoxes / numThreads;	
		if (averageBoxesPerThread == 0) averageBoxesPerThread = 1;
			
		curThreadWeTakeBoxesIndex = 0;
		for (int m = 0; m < numThreads; m++) {
			if (workLen[m] < averageBoxesPerThread) {
				for (int n = curThreadWeTakeBoxesIndex; n < numThreads; n++) {
					if (workLen[n] > averageBoxesPerThread) {	
						numBoxesWeTake = averageBoxesPerThread - workLen[m] <= workLen[n] - averageBoxesPerThread ? averageBoxesPerThread - workLen[m] : workLen[n] - averageBoxesPerThread;
						workLen[n] -= numBoxesWeTake;
						memcpy(boxes + m*SIZE_BUFFER_PER_THREAD*(2*inDim+3) + (workLen[m])*(2*inDim+3), boxes + n*SIZE_BUFFER_PER_THREAD*(2*inDim+3) + (workLen[n])*(2*inDim+3), sizeof(double)*(2*inDim+3)*numBoxesWeTake);
						workLen[m] += numBoxesWeTake;	
						if (workLen[m] == averageBoxesPerThread) {
							curThreadWeTakeBoxesIndex = n;
							break;	
						}
					}		
				}
				
			}		
		}
			
		for (int m = 0; m < numThreads; m++) {
			if (workLen[m] == averageBoxesPerThread) {
				for (int n = curThreadWeTakeBoxesIndex; n < numThreads; n++) {
					if (workLen[n] > averageBoxesPerThread + 1) {
						numBoxesWeTake = 1;
						workLen[n] -= numBoxesWeTake;
						memcpy(boxes + m*SIZE_BUFFER_PER_THREAD*(2*inDim+3) + (workLen[m])*(2*inDim+3), boxes + n*SIZE_BUFFER_PER_THREAD*(2*inDim+3) + (workLen[n])*(2*inDim+3), sizeof(double)*(2*inDim+3)*numBoxesWeTake);
						workLen[m] += numBoxesWeTake;
						curThreadWeTakeBoxesIndex = n;						
						break;
					}		
				}	
			}
			if (curThreadWeTakeBoxesIndex == numThreads - 1 && workLen[curThreadWeTakeBoxesIndex] <= averageBoxesPerThread + 1) break;
		}
	}
	auto endCPU = std::chrono::high_resolution_clock::now();
	
	CHECKED_CALL(cudaEventDestroy(start));
	CHECKED_CALL(cudaEventDestroy(stop));
	CHECKED_CALL(cudaFree(dev_inBox));
    CHECKED_CALL(cudaFree(dev_workLen));
	CHECKED_CALL(cudaFree(dev_mins));
	CHECKED_CALL(cudaFree(dev_workCounts));
	
	std::cout << "EPS = " << inEps << "\n";
	std::cout << "MIN = " << funcMin << "\n";
	std::cout << "wc = " << wc << "\n";
	std::cout << "timeAll = " << timeAll << "\n";
	std::cout << "timeAllCPU = " << (std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU)).count() << "\n";
	
	outfile << inDim << "\t" << inEps << "\t" << timeAll << "\t" << (std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU)).count() << "\t" << wc << "\t" << MAX_GPU_ITER << "\n";
	outfile.close();

}


/**
*	Global optimization with CUDA version 3 (no balancing, calculate the number of real working threads)
*	@param inbox pointer to Box
*	@param inDim number of variables
*	@param inEps accuracy
*	@param outBox the optimal box (not uses)
*	@param outMin the global minimum
*	@param status the status of finishing (not uses)
*	@param funRecord initial value of function record
*	@param filename the file name to write results
*/
void fnGetOptValueWithCUDA_v3(double *inBox, const int inDim, const double inEps, double *outBox, double*outMin, int *status, double funRecord, char* filename)
{
	int numBoxes = BLOCK_SIZE*NUM_BLOCKS;
	double *boxes =  new double[numBoxes*(inDim*2+3)*SIZE_BUFFER_PER_THREAD];
	double h;
	int hInd;
	int *workLen;
	double *mins;
	int i,n;	
	double funcMin = funRecord;
	int realWorkingThreadsNumber;

	*status = 1;

	workLen = new int[numBoxes];
	mins = new double[numBoxes];

	h = inBox[1] - inBox[0];
	hInd = 0;
	
	std::ofstream outfile;
	outfile.open(filename,std::ios_base::app);
	if(outfile.fail())
		throw std::ios_base::failure(std::strerror(errno));
		
	for (i = 0; i < inDim; i++) {
		if(h < inBox[i*inDim + 1] - inBox[i*inDim]) {
			h = inBox[i*inDim + 1] - inBox[i*inDim];
			hInd = i;
		}
	}

	for (n = 0; n < numBoxes; n++) {
		for(i = 0; i < inDim; i++) {
			if (i == hInd) {
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2] = inBox[i*2] + h/numBoxes*n;
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2 + 1] = inBox[i*2] + h/numBoxes*(n+1);
			}
			else {
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2] = inBox[i*2];
				boxes[n*(2*inDim + 3)*SIZE_BUFFER_PER_THREAD + i*2 + 1] = inBox[i*2 + 1];
			}
		}
	}
	
	double *dev_inBox = 0;
	int *dev_workLen = 0;
	long long *dev_workCounts = 0;
	double *dev_mins = 0;

	int GridSize = NUM_BLOCKS;
	int numThreads = numBoxes;
	int sizeInBox = numThreads*(inDim*2+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD;
	
	float time, timeAll;
	long long *workCounts = new long long[numThreads*sizeof(int)];
	
	for (i = 0; i < numThreads; i++) {
		workLen[i] = 1;
		workCounts[i] = 0;
	}

	cudaEvent_t start, stop;
	
	CHECKED_CALL(cudaSetDevice(DEVICE));
	CHECKED_CALL(cudaDeviceReset());	
    CHECKED_CALL(cudaMalloc((void **)&dev_inBox, sizeInBox));
	CHECKED_CALL(cudaMalloc((void **)&dev_workLen, numThreads*sizeof(int)));
	CHECKED_CALL(cudaMalloc((void **)&dev_mins, numThreads*sizeof(double)));
	CHECKED_CALL(cudaMalloc((void **)&dev_workCounts, numThreads*sizeof(long long)));
	
	timeAll = 0;
	long long wc = 0;
	long long ls = 0;

	auto startCPU = std::chrono::high_resolution_clock::now();

	CHECKED_CALL(cudaEventCreate(&start));
	CHECKED_CALL(cudaEventCreate(&stop));
	
	for (i = 0; i < MAX_NUM_RUNS ; i++) {
		CHECKED_CALL(cudaMemcpy(dev_inBox, boxes, numBoxes*(2*inDim+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD, cudaMemcpyHostToDevice));
		CHECKED_CALL(cudaMemcpy(dev_workLen, workLen, numThreads*sizeof(int), cudaMemcpyHostToDevice));
		CHECKED_CALL(cudaMemcpy(dev_workCounts, workCounts, numThreads*sizeof(int), cudaMemcpyHostToDevice));
		CHECKED_CALL(cudaEventRecord(start, 0));
		
		globOptCUDA_v1<<<GridSize, BLOCK_SIZE>>>(dev_inBox,inDim,dev_workLen,dev_mins,funcMin,inEps,dev_workCounts);

		CHECKED_CALL(cudaGetLastError());
		CHECKED_CALL(cudaEventRecord(stop, 0));
		CHECKED_CALL(cudaEventSynchronize(stop));
		CHECKED_CALL(cudaDeviceSynchronize());

		CHECKED_CALL(cudaMemcpy(boxes, dev_inBox, numBoxes*(2*inDim+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD, cudaMemcpyDeviceToHost));
		CHECKED_CALL(cudaMemcpy(workLen, dev_workLen, numThreads*sizeof(int), cudaMemcpyDeviceToHost));
		CHECKED_CALL(cudaMemcpy(mins, dev_mins, numThreads*sizeof(double), cudaMemcpyDeviceToHost));
		CHECKED_CALL(cudaMemcpy(workCounts, dev_workCounts, numThreads*sizeof(long long), cudaMemcpyDeviceToHost));

		CHECKED_CALL(cudaEventElapsedTime(&time, start, stop)); 
		
		ls = 0;	
		realWorkingThreadsNumber = 0;
		for (int j = 0; j < numThreads; j++) {
			wc+=workCounts[j];
			ls += workLen[j];
			if (workLen[j] > 0) realWorkingThreadsNumber++;
			workCounts[j] = 0;
		}

		funcMin = mins[0];		
		for (int j  = 1; j < numThreads; j++) {
			if(funcMin > mins[j]) funcMin = mins[j];
		}
	
		timeAll += time;
		if(ls ==0) break;
		
		outfile << inDim << "\t" << inEps << "\t" << "\t" << realWorkingThreadsNumber << "\t" << MAX_GPU_ITER << "\n";
	}
	auto endCPU = std::chrono::high_resolution_clock::now();
	
	CHECKED_CALL(cudaEventDestroy(start));
	CHECKED_CALL(cudaEventDestroy(stop));
	CHECKED_CALL(cudaFree(dev_inBox));
    CHECKED_CALL(cudaFree(dev_workLen));
	CHECKED_CALL(cudaFree(dev_mins));
	CHECKED_CALL(cudaFree(dev_workCounts));
	
	std::cout << "EPS = " << inEps << "\n";
	std::cout << "MIN = " << funcMin << "\n";
	std::cout << "wc = " << wc << "\n";
	std::cout << "timeAll = " << timeAll << "\n";
	std::cout << "timeAllCPU = " << (std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU)).count() << "\n";
	
	outfile.close();

}



































































#endif
