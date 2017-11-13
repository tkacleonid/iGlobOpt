#ifndef __CUDAGLOBALOPTIMIZATION_H__
#define __CUDAGLOBALOPTIMIZATION_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_device_runtime_api.h>
#include "CPUGlobalOptimization.h"

#include <math_functions_dbl_ptx3.h>
#include "interval.h"
#include <stdio.h>
#include <stdlib.h>

// Send Data to GPU to calculate limits
void sendDataToCuda(double *outLimits, const double *inBox, int inRank, int inFunc, int numBoxes);

/**
*	Calculus Interval for Multiple function on GPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsMultiple2_CUDA(double *inBox, int inRank, double *outLimits);
/**
*	Calculus Interval for Hyperbolic function on GPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsHypebolic2_CUDA(double *inBox, int inRank, double *outLimits);
/**
*	Calculus Interval for AluffiPentini function
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsAluffiPentini2_CUDA(double *inBox, int inRank, double *outLimits);
/**
*	Calculus Valuefor Multiple function on GPU
*	@param inbox pointer to point
*	@param inRank number of variables
*	@return value of function
*/
__device__ double fnCalcFunRozenbroke_CUDA(double *inBox, int inRank);
/**
*	Calculus Interval for Rozenbroke function
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsRozenbroke_CUDA(double *inBox, int inRank);
/**
*	Calculus minimum value for function on GPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param inNumBoxesSplitCoeff number of parts for each dimension
*	@param inEps required accuracy
*	@param inMaxIter maximum count of iterations
*	@param inFun number of optimazing function
*	@param outBox pointer to optimal box
*	@param outMin pointer to optimal value
*	@param outEps pointer to reached accuracy
*	@param outEps pointer to status of solving optimization problem
*/
void fnGetOptValueWithCUDA(double *inBox, int inRank, int inNumBoxesSplitCoeff, double inEps, int inMaxIter, int inFunc, double *outBox, double*outMin, double *outEps,int *status);

/**
*	Send data into GPU
*	@param outLimits pointer to calculated limits
*	@param inBox pointer to boxes
*	@param inRank demnsion
*	@param inFunc number of optimazion function
*	@param inNumBoxes number of boxes
*/
void sendDataToCuda(double *outLimits, const double *inBox, int inRank, int inFunc, int inNumBoxes);

void sendDataToCuda_deep( double *inBox, int inRank, int inFunc, int inNumBoxes, int * workLen,double* mins,double inFuncMin);

__global__ void globOptCUDA(double *inBox, int inRank, int *workLen, double *min, double inRec, double inEps);


__global__ void calculateLimitsOnCUDA(double *outLimits, const double *inBox,int inRank,int inFunc, int numBoxes);

__device__ double fnCalcFunMultiple2_CUDA(double *inBox, int inRank)
{
	return inBox[0]*inBox[1];
}

/**
*	Calculus Interval for Multiple function on GPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsMultiple2_CUDA(double *inBox, int inRank, double *outLimits)
{
	
	double x1 = (inBox[0]+inBox[1])/2;
	double x2 = (inBox[2]+inBox[3])/2;

	double var1 = inBox[0]*inBox[2];
	double var2 = inBox[0]*inBox[3];
	double var3 = inBox[1]*inBox[2];
	double var4 = inBox[1]*inBox[3];

	outLimits[0] = fmin(fmin(var1,var2),fmin(var3,var4));
	outLimits[1] = fmax(fmax(var1,var2),fmax(var3,var4));
	outLimits[2] = x1*x2;
}


/**
*	Calculus Interval for Hyperbolic function on GPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsHypebolic2_CUDA(double *inBox, int inRank, double *outLimits)
{
	double limits[2];
	double limits2[2];

	double x1 = (inBox[0]+inBox[1])/2;
	double x2 = (inBox[2]+inBox[3])/2;

	double var1 = inBox[0]*inBox[0];
	double var2 = inBox[0]*inBox[1];
	double var3 = inBox[1]*inBox[1];

	limits[0] = var2 < 0 ? 0 : fmin(fmin(var1,var2),var3);
	limits[1] = fmax(fmax(var1,var2),var3);

	var1 = inBox[2]*inBox[2];
	var2 = inBox[2]*inBox[3];
	var3 = inBox[3]*inBox[3];

	limits2[0] = var2 < 0 ? 0 : fmin(fmin(var1,var2),var3);
	limits2[1] = fmax(fmax(var1,var2),var3);

	outLimits[0] = limits[0] - limits2[1];
	outLimits[1] = limits[1] - limits2[0];
	outLimits[2] = x1*x1-x2*x2;
}

/**
*	Calculus Interval for AluffiPentini function
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/

__device__ void fnCalcFunLimitsAluffiPentini2_CUDA(double *inBox, int inRank, double *outLimits)
{

	double limits[2];
	double limits2[2];

	double x1 = (inBox[0]+inBox[1])/2;
	double x2 = (inBox[2]+inBox[3])/2;

	double var1 = inBox[0]*inBox[0];
	double var2 = inBox[0]*inBox[1];
	double var3 = inBox[1]*inBox[1];

	limits[0] = var2 < 0 ? 0 : fmin(fmin(var1,var2),var3);
	limits[1] = fmax(fmax(var1,var2),var3);

	var1 = inBox[2]*inBox[2];
	var2 = inBox[2]*inBox[3];
	var3 = inBox[3]*inBox[3];

	limits2[0] = limits[0]*limits[0];
	limits2[1] = limits[1]*limits[1];

	outLimits[0] = 0.25*limits2[0] - 0.5*limits[1] + 0.1*inBox[0] + 0.5*(var2 < 0 ? 0 : fmin(fmin(var1,var2),var3));
	outLimits[1] = 0.25*limits2[1] - 0.5*limits[0] + 0.1*inBox[1] + 0.5*(fmax(fmax(var1,var2),var3));
	outLimits[2] = 0.25*pow(x1,4.0)-0.5*pow(x1,2.0) + 0.1*x1 + 0.5*pow(x2,2.0);
}


__device__ double fnCalcFunRozenbroke_CUDA(double *inBox, int inRank)
{
	int i;
	double val = 0;;
	for(i = 0; i < inRank - 1; i++)
	{
		val += ((1 - inBox[i])*(1 - inBox[i]) + 100.0*(inBox[i+1]-inBox[i]*inBox[i])*(inBox[i+1]-inBox[i]*inBox[i]));
	}
	return val;
}



/**
*	Calculus Interval for Rozenbroke function
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsRozenbroke_CUDA(double *inBox, int inRank)
{
	double sup = 0;
	double sub = 0;
	double sup1,sub1,sup2,sub2,a,b,val = 0,var1,var2,var3,x1,x2;
	int i,j;

	for(i = 0; i < 1; i++)
	{
		sub1 = 1 - inBox[i*2 + 1];
		sup1 = 1 - inBox[i*2];
		

		var1 = sup1*sup1;
		var2 = sup1*sub1;
		var3 = sub1*sub1;
		
		sub1 = (sub1*sup1 < 0) ? 0 : fmin(fmin(var1,var2),var3);
		sup1 = fmax(fmax(var1,var2),var3);
		

		var1 = inBox[i*2 + 1]*inBox[i*2 + 1];
		var2 = inBox[i*2 + 1]*inBox[i*2];
		var3 = inBox[i*2]*inBox[i*2];
		
		inBox[inRank*2] = var1;
		inBox[inRank*2 + 1] = var2;
		inBox[inRank*2 + 2] = var3;

		a = (inBox[i*2 + 1]*inBox[i*2] < 0) ? 0 : fmin(fmin(var1,var2),var3);
		b = fmax(fmax(var1,var2),var3);
		
		inBox[inRank*2] = a;
		inBox[inRank*2 + 1] = b;
		inBox[inRank*2 + 2] = var3;

		sub2 = inBox[(i+1)*2] - b;
		sup2 = inBox[(i+1)*2 + 1] - a;
		
		inBox[inRank*2] = sub2;
		inBox[inRank*2 + 1] = sup2;
		inBox[inRank*2 + 2] = var3;

		var1 = sup2*sup2;
		var2 = sup2*sub2;
		var3 = sub2*sub2;
		
		inBox[inRank*2] = var1;
		inBox[inRank*2 + 1] = var2;
		inBox[inRank*2 + 2] = var3;

		sub2 = (sub2*sup2 < 0) ? 0 : 100*fmin(fmin(var1,var2),var3);
		sup2 = 100*fmax(fmax(var1,var2),var3);
		
		//inBox[inRank*2] = sub2;
		//inBox[inRank*2 + 1] = sup2;
		//inBox[inRank*2 + 2] = var3;

		sub += sub1 + sub2;
		sup += sup1 + sup2;

		x1 = (inBox[i*2 + 1] + inBox[i*2])/2;
		x2 = (inBox[(i+1)*2 + 1] + inBox[(i+1)*2])/2;
		val += ((1 - x1)*(1 - x1) + 100*(x2-x1*x1)*(x2-x1*x1));
	}

	
	double x[10];
	//x = (double *) malloc(inRank*sizeof(double));
	double minFun;


	for(j = 0; j < inRank; j++){
			x[j] = (inBox[j*2]+inBox[j*2+1])/2.0;
	}
	minFun = fnCalcFunRozenbroke_CUDA(x, inRank);

	

	inBox[inRank*2] = sub;
	inBox[inRank*2 + 1] = sup;
	inBox[inRank*2 + 2] = val;
	
}



// Send Data to GPU to calculate limits
void sendDataToCuda_deep(double *inBox, int inRank, int inFunc, int numBoxes, int * workLen,double* mins, double funcMin)
{
    double *dev_inBox = 0;
	int *dev_workLen = 0;
	double *dev_mins = 0;

	int GridSize = 1;
	int numThreads = 1024;
	int sizeInBox = numThreads*(inRank*2+3)*sizeof(double)*1024;

	cudaEvent_t start, stop;

	CHECKED_CALL(cudaSetDevice(DEVICE));
	
	CHECKED_CALL(cudaDeviceReset());
	
	std::cout << "start CUDA malloc 1\n";
	
	std::cout << "size CUDA malloc 1" << sizeInBox << "\n";
	
    CHECKED_CALL(cudaMalloc((void **)&dev_inBox, sizeInBox));
	
	std::cout << "start CUDA malloc 2\n";
	
	CHECKED_CALL(cudaMalloc((void **)&dev_workLen, numThreads*sizeof(int)));
	
	std::cout << "start CUDA malloc 3\n";
	
	CHECKED_CALL(cudaMalloc((void **)&dev_mins, numThreads*sizeof(double)));
    CHECKED_CALL(cudaEventCreate(&start));
    CHECKED_CALL(cudaEventCreate(&stop));
	CHECKED_CALL(cudaMemcpy(dev_inBox, inBox, numBoxes*(2*inRank+3)*sizeof(double)*1024, cudaMemcpyHostToDevice));
	//CHECKED_CALL(cudaMemcpy(dev_workLen, workLen, numBoxes*sizeof(int), cudaMemcpyHostToDevice));

	CHECKED_CALL(cudaEventRecord(start, 0));
	std::cout << "call CUDA\n";
	globOptCUDA<<<GridSize, 1024>>>(dev_inBox, inRank,dev_workLen,dev_mins,funcMin, 0.001);
	std::cout << "stop CUDA\n";
    CHECKED_CALL(cudaGetLastError());

    CHECKED_CALL(cudaEventRecord(stop, 0));
    CHECKED_CALL(cudaDeviceSynchronize());

    CHECKED_CALL(cudaMemcpy(inBox, dev_inBox, numBoxes*(2*inRank+3)*sizeof(double)*1024, cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(workLen, dev_workLen, numBoxes*sizeof(int), cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(mins, dev_mins, numBoxes*sizeof(double), cudaMemcpyDeviceToHost));

	float time;
    CHECKED_CALL(cudaEventElapsedTime(&time, start, stop));

	std::cout << "time = " << time << "\t"  << "numBoxes = " << numBoxes << "\n";

    CHECKED_CALL(cudaEventDestroy(start));
    CHECKED_CALL(cudaEventDestroy(stop));
	
	std::cout << "free start\n";
	
	CHECKED_CALL(cudaFree(dev_inBox));
	
	std::cout << "free 1\n";
	
    CHECKED_CALL(cudaFree(dev_workLen));
	
	std::cout << "free 2\n";
	
	CHECKED_CALL(cudaFree(dev_mins));
	
	std::cout << "free 3\n";
	
	std::cout <<  "\n\n\n";
	
	for(int i  = 0; i < 1024; i++)
	{
		std::cout << workLen[i] << "\t";
	}
	
	std::cout <<  "\n\n\n";
	
	for(int i  = 102400; i < 102410; i++)
	{
		for(int j = 0; j < inRank; j++)
		{
			std::cout << "[" << inBox[i*(2*inRank + 3) + j*2] << "; " << inBox[i*(2*inRank + 3) + j*2 + 1] << "]\t";
		}
		std::cout << "\t\t" << inBox[i*(2*inRank + 3) + inRank*2] << "\t\t" << inBox[i*(2*inRank + 3) + inRank*2 + 1] << "\t\t" << inBox[i*(2*inRank + 3) + inRank*2 + 2];
		std::cout <<  "\n";
	}
	
	std::cout <<  "\n\n\n";
	
	std::cout << "MIN = " << mins[0] << "\n\n\n";
	
	for(int i  = 0; i < 1024; i++)
	{
		std::cout << mins[i] << "\t";
	}
	
	std::cout <<  "\n\n\n";
    

}

void fnGetOptValueWithCUDA_deep(double *inBox, int inRank, int inNumBoxesSplitCoeff, double inEps, int inMaxIter, int inFunc, double *outBox, double*outMin, double *outEps,int *status)
{
	int numBoxes = 1024;
	double *boxes =  new double[numBoxes*(inRank*2+3)*1024];
	double h;
	int hInd;
	int *workLen;
	double *mins;

	int i,n;
	
	double funcMin = 0;

	funcMin = 0;

	*status = 1;


	workLen = new int[numBoxes];
	mins = new double[numBoxes];

	h = inBox[1] - inBox[0];
	hInd = 0;
	
	std::cout << "start finding max width rank\n";
	for(i = 0; i < inRank; i++)
	{
		if(h < inBox[i*inRank + 1] - inBox[i*inRank])
		{
			h = inBox[i*inRank + 1] - inBox[i*inRank];
			hInd = i;
		}
	}
	std::cout << "stop finding max width rank\n";

	for(n = 0; n < numBoxes; n++)
	{
		for(i = 0; i < inRank; i++)
		{
			if(i == hInd)
			{
				boxes[n*(2*inRank + 3)*1024 + i*2] = inBox[i*2] + h/1024.0*n;
				boxes[n*(2*inRank + 3)*1024 + i*2 + 1] = inBox[i*2] + h/1024.0*(n+1);
			}
			else
			{
				boxes[n*(2*inRank + 3)*1024 + i*2] = inBox[i*2];
				boxes[n*(2*inRank + 3)*1024 + i*2 + 1] = inBox[i*2 + 1];
			}
		}

	}
	
	std::cout << "send data\n";
	sendDataToCuda_deep(boxes, inRank, inFunc, numBoxes, workLen,mins,funcMin);

}


__const__ double rank = 10;


__global__ void globOptCUDA(double *inBox, int inRank, int *workLen, double *min, double inRec, double inEps)
{
	__shared__ double min_s[1024];
	__shared__ int workLen_s[1024];
	__shared__ int workLen_s_temp[1024];
	__shared__ int count[1024];
	
	double minRec = inRec;
	int i, j, bInd, hInd;
	double curEps, h;
	
	int threadId = blockIdx.x * 1024 + threadIdx.x;
	
	workLen_s[threadId] = 1;
	min_s[threadId] = minRec;
	
	count[threadId] = 0;
	
	int wl;
	
	inEps = 0.000001;
	
	__syncthreads();
	
	while(workLen_s[threadId] < 1024 && count[threadId] < 1000000)
	{
		if(workLen_s[threadId] > 0)
		{
			
			bInd = threadId*1024*(2*inRank+3) + (workLen_s[threadId] - 1)*(2*inRank+3);
			fnCalcFunLimitsRozenbroke_CUDA(inBox + bInd, inRank);
			
			if(min_s[threadId] > inBox[bInd + 2*inRank + 2])
			{
				min_s[threadId] = inBox[bInd + 2*inRank + 2];
			}

			curEps = min_s[threadId] - inBox[bInd + 2*inRank];
			//curEps = curEps > 0 ? curEps : -curEps;	
			
			
			if(min_s[threadId] - inEps < inBox[bInd + 2*inRank])
			{
				--workLen_s[threadId];
			}
			else
			{
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
				++workLen_s[threadId];
			}
			
		}

		__syncthreads();
		for(i = 0; i < 1024; i++)
		{
			if(minRec > min_s[blockIdx.x * 1024 + i])
			{
				minRec = min_s[blockIdx.x * 1024 + i];
			}
		}
		__syncthreads();
		min_s[threadId] = minRec;		
		
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
			
			
		if(threadId == 0 && (count[threadId]+1) % 100 == 0)
		{
			for(i = 1; i < 1024; i++)
			{
				if(workLen_s[i] == 0)
				{
					for(j = 0; j < 1024; j++)
					{
						if(workLen_s[j] > 4)
						{
							atomicAdd(workLen_s + j, -2);
							memcpy(inBox + i*1024*(2*inRank+3), inBox + j*1024*(2*inRank+3) + (workLen_s[j] - 1)*(2*inRank+3), sizeof(double)*(2*inRank+3)*2);
							workLen_s[i] += 2;
							break;
						}
					}
				}
			}
		}	
			
			
			
			
			
		
		__syncthreads();
		
		wl = workLen_s[0];
		for(i = 0; i < 1024; i++)
		{
			if(wl < workLen_s[i]) wl = workLen_s[i];
		}
		if(wl == 0) break;
		
		++count[threadId];

		
	}
	workLen[threadId] = workLen_s[threadId];
	min[threadId] = minRec;
	
}


#endif