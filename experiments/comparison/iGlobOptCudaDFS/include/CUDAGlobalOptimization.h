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

__global__ void globOptCUDA_1(double *inBox, double *droppedBoxes, int inRank, int *workLen, double *min, double inRec, double inEps, long long *workCounts);


__global__ void globOptCUDA_2(double *inBox, int inRank, int *workLen, double *min, double inRec, double inEps, long long *workCounts);



/**
*	Calculus Interval for Rozenbroke function
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsStyblinski_CUDA(double *inBox, int inRank)
{
	double sup = 0;
	double sub = 0;
	double sup1,sub1,sup2,sub2,val = 0,var1,var2,var3,x;
	int i;

	for(i = 0; i < inRank; i++)
	{
			
		var1 = inBox[i*2 + 1]*inBox[i*2 + 1];
		var2 = inBox[i*2 + 1]*inBox[i*2];
		var3 = inBox[i*2]*inBox[i*2];
		
		
		
		
		sub1 = fmin(fmin(var1,var2),var3);
		sup1 = fmax(fmax(var1,var2),var3);
		
		var1 = sub1*sub1;
		var2 = sub1*sup1;
		var3 = sup1*sup1;
		
		sub2 = fmin(fmin(var1,var2),var3);
		sup2 = fmax(fmax(var1,var2),var3);

		sub += (sub2 - 16*sup1 + 5*fmin(inBox[i*2 + 1],inBox[i*2]))/2.0;
		sup += (sup2 - 16*sub1 + 5*fmax(inBox[i*2 + 1],inBox[i*2]))/2.0;
		
		
		
		

		//sub += sub1;
		//sup += sup1;

		x = (inBox[i*2 + 1] + inBox[i*2])/2;
		val += (x*x*x*x - 16*x*x + 5*x)/2.0;
			
	}
	

	inBox[2*inRank + 0] = sub;
	inBox[2*inRank + 1] = sup;
	inBox[2*inRank+2] = val;
	
}



void fnGetOptValueWithCUDA(double *inBox, const int inRank, const double inEps, double *outBox, double*outMin, int *status)
{
	int numBoxes = BLOCK_SIZE*NUM_BLOCKS;
	double *boxes =  new double[numBoxes*(inRank*2+3)*SIZE_BUFFER_PER_THREAD];
	double *droppedBoxes =  new double[numBoxes*(inRank*2+3)];
	double h;
	int hInd;
	int *workLen;
	double *mins;

	int i,n;
	
	double funcMin = 0;

	funcMin = -39.1661657038*inRank;

	*status = 1;

	workLen = new int[numBoxes];
	mins = new double[numBoxes];

	h = inBox[1] - inBox[0];
	hInd = 0;
	
	
	
	for(i = 0; i < inRank; i++)
	{
		if(h < inBox[i*inRank + 1] - inBox[i*inRank])
		{
			h = inBox[i*inRank + 1] - inBox[i*inRank];
			hInd = i;
		}
	}

	for(n = 0; n < numBoxes; n++)
	{
		for(i = 0; i < inRank; i++)
		{
			if(i == hInd)
			{
				boxes[n*(2*inRank + 3)*SIZE_BUFFER_PER_THREAD + i*2] = inBox[i*2] + h/numBoxes*n;
				boxes[n*(2*inRank + 3)*SIZE_BUFFER_PER_THREAD + i*2 + 1] = inBox[i*2] + h/numBoxes*(n+1);
			}
			else
			{
				boxes[n*(2*inRank + 3)*SIZE_BUFFER_PER_THREAD + i*2] = inBox[i*2];
				boxes[n*(2*inRank + 3)*SIZE_BUFFER_PER_THREAD + i*2 + 1] = inBox[i*2 + 1];
			}
		}

	}
	
	
	for (int j = 0; j < numBoxes; j++) {
		for (int k = 0; k < inRank; k++) {
			droppedBoxes[j*(2*inRank+3)+2*k] = 0.0;
			droppedBoxes[j*(2*inRank+3)+2*k +1] = 0.0;
				
		}
		droppedBoxes[j*(2*inRank+3)+2*inRank] = 0.0;
		droppedBoxes[j*(2*inRank+3)+2*inRank+1] = 0.0;
		droppedBoxes[j*(2*inRank+3)+2*inRank+2] = 0.0;
	}
	

	double *dev_inBox = 0;
	double *dev_droppedBoxes = 0;
	int *dev_workLen = 0;
	long long *dev_workCounts = 0;
	double *dev_mins = 0;

	int GridSize = NUM_BLOCKS;
	int numThreads = numBoxes;
	int sizeInBox = numThreads*(inRank*2+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD;
	
	float time, timeAll;
	
	long long *workCounts = new long long[numThreads*sizeof(int)];
	
	for(i = 0; i < numThreads; i++)
	{
		workLen[i] = 1;
		workCounts[i] = 0;
	}

	cudaEvent_t start, stop;
	
	std::cout << "in 1\n";
	
	CHECKED_CALL(cudaSetDevice(DEVICE));
	
	
	
	CHECKED_CALL(cudaDeviceReset());
	
	std::cout << "start CUDA malloc 1\n";
		
    CHECKED_CALL(cudaMalloc((void **)&dev_inBox, sizeInBox));
    CHECKED_CALL(cudaMalloc((void **)&dev_droppedBoxes, numThreads*(inRank*2+3)*sizeof(double)));
	
	std::cout << "start CUDA malloc 2\n";
	
	CHECKED_CALL(cudaMalloc((void **)&dev_workLen, numThreads*sizeof(int)));
	
	std::cout << "start CUDA malloc 3\n";
	
	CHECKED_CALL(cudaMalloc((void **)&dev_mins, numThreads*sizeof(double)));
	
	std::cout << "start CUDA malloc 4\n";
	
	CHECKED_CALL(cudaMalloc((void **)&dev_workCounts, numThreads*sizeof(long long)));
	
	timeAll = 0;
	long long wc = 0;
	for(i = 0; i < MAX_NUM_RUNS ; i++)
	{
		std::cin.get();
		std::cout << "\nNUMBER #" << (i+1) << "\n";
		
		CHECKED_CALL(cudaEventCreate(&start));
		CHECKED_CALL(cudaEventCreate(&stop));
		CHECKED_CALL(cudaMemcpy(dev_inBox, boxes, numBoxes*(2*inRank+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD, cudaMemcpyHostToDevice));
		CHECKED_CALL(cudaMemcpy(dev_droppedBoxes, droppedBoxes, numBoxes*(2*inRank+3)*sizeof(double), cudaMemcpyHostToDevice));
		CHECKED_CALL(cudaMemcpy(dev_workLen, workLen, numThreads*sizeof(int), cudaMemcpyHostToDevice));
		CHECKED_CALL(cudaMemcpy(dev_workCounts, workCounts, numThreads*sizeof(int), cudaMemcpyHostToDevice));
		CHECKED_CALL(cudaEventRecord(start, 0));
		std::cout << "call CUDA\n";
		switch(TYPE_CUDA_OPTIMIZATION)
		{
			case 1:
				globOptCUDA_1<<<GridSize, BLOCK_SIZE>>>(dev_inBox, dev_droppedBoxes,inRank,dev_workLen,dev_mins,funcMin,inEps,dev_workCounts);
				break;
			case 2:
				globOptCUDA_2<<<GridSize, BLOCK_SIZE>>>(dev_inBox, inRank,dev_workLen,dev_mins,funcMin,inEps,dev_workCounts);
				break;
				
				
			default:
				globOptCUDA_1<<<GridSize, BLOCK_SIZE>>>(dev_inBox, dev_droppedBoxes,inRank,dev_workLen,dev_mins,funcMin,inEps,dev_workCounts);
				break;
		}
		
		std::cout << "stop CUDA\n";
		CHECKED_CALL(cudaGetLastError());

		CHECKED_CALL(cudaEventRecord(stop, 0));
		CHECKED_CALL(cudaEventSynchronize(stop));
		CHECKED_CALL(cudaDeviceSynchronize());

		CHECKED_CALL(cudaMemcpy(boxes, dev_inBox, numBoxes*(2*inRank+3)*sizeof(double)*SIZE_BUFFER_PER_THREAD, cudaMemcpyDeviceToHost));
		CHECKED_CALL(cudaMemcpy(droppedBoxes, dev_droppedBoxes, numBoxes*(2*inRank+3)*sizeof(double), cudaMemcpyDeviceToHost));
		CHECKED_CALL(cudaMemcpy(workLen, dev_workLen, numThreads*sizeof(int), cudaMemcpyDeviceToHost));
		CHECKED_CALL(cudaMemcpy(mins, dev_mins, numThreads*sizeof(double), cudaMemcpyDeviceToHost));
		CHECKED_CALL(cudaMemcpy(workCounts, dev_workCounts, numThreads*sizeof(long long), cudaMemcpyDeviceToHost));

		CHECKED_CALL(cudaEventElapsedTime(&time, start, stop)); 
		

		std::cout << "time = " << time << "\n";
		
		long long ls = 0;
		
		for(int j = 0; j < numThreads; j++)
		{
			wc+=workCounts[j];
			ls += workLen[j];
			workCounts[j] = 0;
		}
		
		std::cout << "wc = " << wc << "\n";
		std::cout << "ls = " << ls << "\n";
		
		funcMin = mins[0];		
		for(int j  = 1; j < numThreads; j++)
		{
			if(funcMin > mins[j]) funcMin = mins[j];
		}
		
		printf("mins: %.10f\n",funcMin);
		printf("\n\n\n");
		
		for (int j = 0; j < numThreads; j++) {
			for (int k = 0; k < inRank; k++) {
				printf("[%f; %f]\t",droppedBoxes[j*(2*inRank+3)+2*k], droppedBoxes[j*(2*inRank+3)+2*k + 1]);
				droppedBoxes[j*(2*inRank+3)+2*k] = 0.0;
				droppedBoxes[j*(2*inRank+3)+2*k +1] = 0.0;
				
			}
			printf("%f\t%f\t%f\n",droppedBoxes[j*(2*inRank+3)+2*inRank], droppedBoxes[j*(2*inRank+3)+2*inRank + 1], droppedBoxes[j*(2*inRank+3)+2*inRank+2]);
			droppedBoxes[j*(2*inRank+3)+2*inRank] = 0.0;
			droppedBoxes[j*(2*inRank+3)+2*inRank+1] = 0.0;
			droppedBoxes[j*(2*inRank+3)+2*inRank+2] = 0.0;
		}
			
		
		
		printf("\n\n\n");
		
		
			
	
		int numWorkBoxes = 0;
		int averageBoxesPerThread = 0;
		int curThreadWeTakeBoxesIndex = -1;
		int curThreadWeTakeBoxesCount = 0;
		int numBoxesWeTake = 0;
		int boxIndex = 0;

		for(int m = 0; m < numThreads; m++)
		{
			numWorkBoxes += workLen[m]; 	
		}
		averageBoxesPerThread = numWorkBoxes / numThreads;
			
		if(averageBoxesPerThread == 0) averageBoxesPerThread = 1;
			
		curThreadWeTakeBoxesIndex = 0;
		for(int m = 0; m < numThreads; m++)
		{
			if(workLen[m] < averageBoxesPerThread)
			{
				for(int n = curThreadWeTakeBoxesIndex; n < numThreads; n++)
				{
					if(workLen[n] > averageBoxesPerThread)
					{
							
						numBoxesWeTake = averageBoxesPerThread - workLen[m] <= workLen[n] - averageBoxesPerThread ? averageBoxesPerThread - workLen[m] : workLen[n] - averageBoxesPerThread;
						workLen[n] -= numBoxesWeTake;
						memcpy(boxes + m*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen[m])*(2*inRank+3), boxes + n*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen[n])*(2*inRank+3), sizeof(double)*(2*inRank+3)*numBoxesWeTake);
						workLen[m] += numBoxesWeTake;	
						if(workLen[m] == averageBoxesPerThread) 
						{
							curThreadWeTakeBoxesIndex = n;
							break;	
						}
					}
						
				}
				
			}
				
		}
			

		for(int m = 0; m < numThreads; m++)
		{
			if(workLen[m] == averageBoxesPerThread)
			{
				for(int n = curThreadWeTakeBoxesIndex; n < numThreads; n++)
				{
					if(workLen[n] > averageBoxesPerThread + 1)
					{
						numBoxesWeTake = 1;
						workLen[n] -= numBoxesWeTake;
						memcpy(boxes + m*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen[m])*(2*inRank+3), boxes + n*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen[n])*(2*inRank+3), sizeof(double)*(2*inRank+3)*numBoxesWeTake);
						workLen[m] += numBoxesWeTake;
						curThreadWeTakeBoxesIndex = n;						
						break;
					}
						
				}
				
			}
			if(curThreadWeTakeBoxesIndex == numThreads - 1 && workLen[curThreadWeTakeBoxesIndex] <= averageBoxesPerThread + 1)
			{
				break;
			}
		}
		 


		CHECKED_CALL(cudaEventDestroy(start));
		CHECKED_CALL(cudaEventDestroy(stop));
		
		timeAll += time;
		if(ls ==0) break;
	}	
	
	
	std::cout << "free start\n";
	
	CHECKED_CALL(cudaFree(dev_inBox));
    CHECKED_CALL(cudaFree(dev_workLen));
	CHECKED_CALL(cudaFree(dev_mins));
	CHECKED_CALL(cudaFree(dev_workCounts));
	
	std::cout << "free stop\n";
	std::cout <<  "\n\n";

	
	std::cout << "MIN = " << funcMin << "\n";
	std::cout <<  "timeAll = " << timeAll;
	std::cout <<  "\n";

}




__global__ void globOptCUDA_1(double *inBox, double *droppedBoxes, int inRank, int *workLen, double *min, double inRec, double inEps, long long *workCounts)

{
	__shared__ double min_s[BLOCK_SIZE];
	__shared__ int workLen_s[BLOCK_SIZE];
	__shared__ int count[BLOCK_SIZE];
	
	double minRec = inRec;
	int i, j,bInd, hInd, n;
	double  h;
	
	int threadId = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	
	workLen_s[threadIdx.x] = workLen[threadId];
	min_s[threadIdx.x] = minRec;
	
	count[threadIdx.x] = 0;
	
	int half;
	
	__syncthreads();	
			
		
	n = 0;
	if(threadId == 0)
	{
		for(i = 0; i < BLOCK_SIZE; i++)
		{
			if(workLen_s[i] == 0)
			{
				for(j = 0; j < BLOCK_SIZE; j++)
				{
					if(workLen_s[j] > BORDER_BALANCE)
					{
						half = workLen_s[j]/2;
						workLen_s[j] -= half;
							memcpy(inBox + (i+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3), inBox + (j+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*half);
							workLen_s[i] += half;
							break;
					}
				}
			}		
		}		
	}	
			

	__syncthreads();
	
	while(workLen_s[threadIdx.x] > 0 && workLen_s[threadIdx.x] < SIZE_BUFFER_PER_THREAD && count[threadIdx.x] < MAX_GPU_ITER)
	{
		
		bInd = threadId*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[threadIdx.x] - 1)*(2*inRank+3);
		fnCalcFunLimitsStyblinski_CUDA(inBox + bInd, inRank);
			
		if(min_s[threadIdx.x] > inBox[bInd + 2*inRank + 2])
		{
			min_s[threadIdx.x] = inBox[bInd + 2*inRank + 2];
		}
			
		if(min_s[threadIdx.x] - inBox[bInd + 2*inRank] < inEps)
		{
			--workLen_s[threadIdx.x];
			memcpy(droppedBoxes+threadId*(2*inRank+3),inBox + bInd,(2*inRank+3)*sizeof(double));
			n++;
		}
		else
		{	
			for(int k = 0; k < 2; k++)
			{
				bInd = threadId*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[threadIdx.x] - 1)*(2*inRank+3);
			hInd = 0;
			h = inBox[bInd + 1] - inBox[bInd];
			for(i = 0; i < inRank; i++)
			{
				if( h < inBox[bInd + i*2 + 1] - inBox[bInd + i*2]) 
				{
					h = inBox[bInd + i*2 + 1] - inBox[bInd + i*2];
					hInd = i;
				}
			}
			for(i = 0; i < inRank; i++)
			{
				if(i == hInd) 
				{
					inBox[bInd + i*2 + 1] = inBox[bInd + i*2] + h/2.0;
					inBox[bInd + 2*inRank + 3 + i*2] = inBox[bInd + i*2] + h/2.0;
					inBox[bInd + 2*inRank + 3 + i*2 + 1] = inBox[bInd + i*2] + h;
				}
				else
				{
					inBox[bInd + 2*inRank + 3 + i*2] = inBox[bInd + i*2];
					inBox[bInd + 2*inRank + 3 + i*2 + 1] = inBox[bInd + i*2 + 1];
				}
			}
			++workLen_s[threadIdx.x];
			}
		}
			
		++count[threadIdx.x];	
	}
	
	workLen[threadId] = workLen_s[threadIdx.x];
	min[threadId] = min_s[threadIdx.x];
	workCounts[threadId]+=n;
	
}


__global__ void globOptCUDA_2(double *inBox, const int inRank, int *workLen, double *min, const double inRec, const double inEps, long long *workCounts)
{
	__shared__ double min_s[BLOCK_SIZE];
	__shared__ int workLen_s[BLOCK_SIZE];
	__shared__ int count[BLOCK_SIZE];
	
	double minRec = inRec;
	int i, j,bInd, hInd, n;
	double  h;
	
	int threadId = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	
	workLen_s[threadIdx.x] = workLen[threadId];
	min_s[threadIdx.x] = minRec;
	
	count[threadIdx.x] = 0;
	
	int half;
	
	
	
	__syncthreads();	
			

	n = 0;
	
	while(workLen_s[threadIdx.x] < BLOCK_SIZE && count[threadIdx.x] < MAX_GPU_ITER)
	{
		if(workLen_s[threadIdx.x] > 0)
		{
			
			bInd = threadId*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[threadIdx.x] - 1)*(2*inRank+3);
			fnCalcFunLimitsStyblinski_CUDA(inBox + bInd, inRank);
			
			if(min_s[threadIdx.x] > inBox[bInd + 2*inRank + 2])
			{
				min_s[threadIdx.x] = inBox[bInd + 2*inRank + 2];
			}
	
			if(min_s[threadIdx.x] - inBox[bInd + 2*inRank] < inEps)
			{
				--workLen_s[threadIdx.x];
				n++;
			}
			else
			{
				

				for(int j = 0; j < 16; j++)
				{
				bInd = threadId*1024*(2*inRank+3) + (workLen_s[threadIdx.x] - 1)*(2*inRank+3);
					
				hInd = 0;
				h = inBox[bInd + 1] - inBox[bInd];
				
				for(i = 0; i < inRank; i++)
				{
					if( h < inBox[bInd + i*2 + 1] - inBox[bInd + i*2]) 
					{
						h = inBox[bInd + i*2 + 1] - inBox[bInd + i*2];
						hInd = i;
					}
				}
				for(i = 0; i < inRank; i++)
				{
					if(i == hInd) 
					{
						inBox[bInd + i*2 + 1] = inBox[bInd + i*2] + h/2.0;
						inBox[bInd + 2*inRank + 3 + i*2] = inBox[bInd + i*2] + h/2.0;
						inBox[bInd + 2*inRank + 3 + i*2 + 1] = inBox[bInd + i*2] + h;
					}
					else
					{
						inBox[bInd + 2*inRank + 3 + i*2] = inBox[bInd + i*2];
						inBox[bInd + 2*inRank + 3 + i*2 + 1] = inBox[bInd + i*2 + 1];
					}
				}
				++workLen_s[threadIdx.x];
				}
			}
		}
			
		

		
		__syncthreads();
		
		if((threadIdx.x == 0))// && (count[threadIdx.x]+1) % 10 == 0)
		{
			for(i = 0; i < BLOCK_SIZE; i++)
			{
				if(minRec > min_s[i])
				{
					minRec = min_s[i];
				}
			}
		}
		
		__syncthreads();
		
		min_s[threadIdx.x] = minRec;	
		
		
		
		/*
		workLen_s_temp[threadId] = workLen[threadId];
		
		__syncthreads();
		
			
		
			
		if(workLen_s[threadId] == 0)
		{
			for(i = 0; i < 1024; i++)
			{
				if(workLen_s_temp[i] > 6 && workLen_s_temp[threadId] == 0)
				{
					atomicAdd(workLen_s_temp + i, -3);
					memcpy(inBox + bInd, inBox + i*1024*(2*inRank+3) + (workLen_s_temp[i] - 1)*(2*inRank+3), sizeof(double)*(2*inRank+3)*3);
					workLen_s_temp[threadId] += 3;
					break;
				}
			}
		}
			
			//workLen[threadId] = workLen_s_temp[threadId];
		__syncthreads();
			
		workLen_s[threadId] = workLen_s_temp[threadId];
			
			*/		
			
		__syncthreads();	
		
		/*
		
		
		if(threadIdx.x == 0 && (count[threadIdx.x]+1) % MAX_ITER_BEFORE_BALANCE == 0)
		{
			for(i = 0; i < BLOCK_SIZE; i++)
			{
				if(workLen_s[i] == 0)
				{
					for(j = 0; j < BLOCK_SIZE; j++)
					{
						if(workLen_s[j] > BORDER_BALANCE)
						{
							half = workLen_s[j]/2;
							workLen_s[j] -= half;
								memcpy(inBox + (i+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3), inBox + (j+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*half);
								workLen_s[i] += half;
								break;
						}
					}
				}		
			}	
		}	
		
		
		
		
		
		
		*/
		
		int numWorkBoxes = 0;
		int averageBoxesPerThread = 0;
		int curThreadWeTakeBoxesIndex = -1;
		int curThreadWeTakeBoxesCount = 0;
		int numBoxesWeTake = 0;
		int boxIndex = 0;
		if(threadIdx.x == 0 && (count[threadIdx.x]+1) % MAX_ITER_BEFORE_BALANCE == 0)
		{
			for(i = 0; i < BLOCK_SIZE; i++)
			{
				numWorkBoxes += workLen_s[i]; 	
			}
			averageBoxesPerThread = numWorkBoxes / BLOCK_SIZE;
			
			if(averageBoxesPerThread == 0) averageBoxesPerThread = averageBoxesPerThread + 1;
			
			curThreadWeTakeBoxesIndex = 0;
			for(i = 0; i < BLOCK_SIZE; i++)
			{
				if(workLen_s[i] < averageBoxesPerThread)
				{
					for(j = curThreadWeTakeBoxesIndex; j < BLOCK_SIZE; j++)
					{
						if(workLen_s[j] > averageBoxesPerThread)
						{
							
							numBoxesWeTake = averageBoxesPerThread - workLen_s[i] <= workLen_s[j] - averageBoxesPerThread ? averageBoxesPerThread - workLen_s[i] : workLen_s[j] - averageBoxesPerThread;
							workLen_s[j] -= numBoxesWeTake;
							memcpy(inBox + (i+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[i])*(2*inRank+3), inBox + (j+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*numBoxesWeTake);
							workLen_s[i] += numBoxesWeTake;	
							if(workLen_s[i] == averageBoxesPerThread) 
							{
								break;	
							}
						}
						
					}
					curThreadWeTakeBoxesIndex = j;
				}
				
			}
			
			
			boxIndex = 0;
			for(i = 0; i < BLOCK_SIZE; i++)
			{
				if(workLen_s[i] == averageBoxesPerThread)
				{
					for(j = curThreadWeTakeBoxesIndex; j < BLOCK_SIZE; j++)
					{
						if(workLen_s[j] > averageBoxesPerThread + 1)
						{
							numBoxesWeTake = 1;
							workLen_s[j] -= numBoxesWeTake;
							memcpy(inBox + (i+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[i])*(2*inRank+3), inBox + (j+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*numBoxesWeTake);
							workLen_s[i] += numBoxesWeTake;	
							break;
						}
						
					}
					curThreadWeTakeBoxesIndex = j;
				}
				if(curThreadWeTakeBoxesIndex == BLOCK_SIZE - 1 && workLen_s[curThreadWeTakeBoxesIndex] <= averageBoxesPerThread + 1)
				{
					break;
				}
			}
			
			
			

			
		}
			
		
		
		__syncthreads();
		
		

		++count[threadIdx.x];

		
	}
	
	workLen[threadId] = workLen_s[threadIdx.x];
	min[threadId] = min_s[threadIdx.x];
	workCounts[threadId]=n;
	
}
























































#endif
