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
__device__ void fnCalcFunLimitsRozenbroke_CUDA(double *inBox, int inRank, double *outLimits);
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

	double a = inBox[0];
	double b = inBox[1];

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

	for(i = 0; i < inRank - 1; i++)
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

		a = (inBox[i*2 + 1]*inBox[i*2] < 0) ? 0 : fmin(fmin(var1,var2),var3);
		b = fmax(fmax(var1,var2),var3);

		sub2 = inBox[(i+1)*2] - b;
		sup2 = inBox[(i+1)*2 + 1] - a;

		var1 = sup2*sup2;
		var2 = sup2*sub2;
		var3 = sub2*sub2;

		sub2 = (sub2*sup2 < 0) ? 0 : 100*fmin(fmin(var1,var2),var3);
		sup2 = 100*fmax(fmax(var1,var2),var3);

		sub += sub1 + sub2;
		sup += sup1 + sup2;

		x1 = (inBox[i*2 + 1] + inBox[i*2])/2;
		x2 = (inBox[(i+1)*2 + 1] + inBox[(i+1)*2])/2;
		val += ((1 - x1)*(1 - x1) + 100*(x2-x1*x1)*(x2-x1*x1));
	}

	
	double *x;
	x = (double *) malloc(inRank*sizeof(double));
	double minFun;


	for(j = 0; j < inRank; j++){
			x[j] = (inBox[j*2]+inBox[j*2+1])/2.0;
	}
	minFun = fnCalcFunRozenbroke_CUDA(x, inRank);


	inBox[inRank*2] = sub;
	inBox[inRank*2 + 1] = sup;
	inBox[inRank*2 + 2] = minFun;
}



void fnGetOptValueWithCUDA(double *inBox, int inRank, int inNumBoxesSplitCoeff, double inEps, int inMaxIter, int inFunc, double *outBox, double*outMin, double *outEps,int *status)
{
	int curNumBoxesSplitCoeff = inNumBoxesSplitCoeff;
	int numBoxes = pow((double) curNumBoxesSplitCoeff,inRank);
	double *boxes =  new double[numBoxes*inRank*2];
	double *boxesResult = new double[numBoxes*3];
	double *restBoxes = new double[inRank*2];
	double *tempRestBoxes = NULL;
	double *h = new double[inRank];
	int numNewBoxes = 0;

	memcpy(restBoxes,inBox,inRank*2*sizeof(double));

	int numRestBoxes = 1;
	int index = 0;
	int i,j,k,n;
	double temp;

	int countIter = 0;
	double curEps = inEps*10;
	
	double funcMin = 0;
	int boxMinIndex = 0;


	*status = 1;
	while((countIter < inMaxIter) && (curEps >= inEps))
	{
		curNumBoxesSplitCoeff = (int) (inNumBoxesSplitCoeff/pow(numRestBoxes,1.0/inRank)) + 2;
		numBoxes = pow((double) curNumBoxesSplitCoeff,inRank);
		boxes = new double[numRestBoxes*numBoxes*inRank*2];
		boxesResult = new double[numRestBoxes*numBoxes*3];
		for(k = 0; k < numRestBoxes; k++)
		{
			for(i = 0; i < inRank; i++)
			{
				h[i] = (restBoxes[(k*inRank+i)*2 + 1] - restBoxes[(k*inRank+i)*2])/curNumBoxesSplitCoeff;
			}

			for(n = 0; n < numBoxes; n++)
			{
				for(i = 0; i < inRank; i++)
				{
					index = ((n % numBoxes) % (long) pow((double)curNumBoxesSplitCoeff,i+1))/((long)pow((double)curNumBoxesSplitCoeff,i));
					boxes[((k*numBoxes + n)*inRank+i)*2] = restBoxes[(k*inRank+i)*2] + h[i]*index;
					boxes[((k*numBoxes + n)*inRank+i)*2 + 1] = restBoxes[(k*inRank+i)*2] + h[i]*(index+1);
				}
			}
		}

		sendDataToCuda(boxesResult, boxes, inRank, inFunc, numRestBoxes*numBoxes);

		funcMin = boxesResult[2];
		boxMinIndex = 0;
		for(n = 0; n < numRestBoxes*numBoxes; n++)
		{
			for(i = n + 1; i < numRestBoxes*numBoxes; i++)
			{
				if(boxesResult[n*3] > boxesResult[i*3])
				{
					temp = boxesResult[n*3];
					boxesResult[n*3] = boxesResult[i*3];
					boxesResult[i*3] = temp;

					temp = boxesResult[n*3+1];
					boxesResult[n*3+1] = boxesResult[i*3+1];
					boxesResult[i*3+1] = temp;

					temp = boxesResult[n*3+2];
					boxesResult[n*3+2] = boxesResult[i*3+2];
					boxesResult[i*3+2] = temp;

					for(j=0; j < inRank; j++)
					{
						temp = boxes[(n*inRank+j)*2];
						boxes[(n*inRank+j)*2] = boxes[(i*inRank+j)*2];
						boxes[(i*inRank+j)*2] = temp;

						temp = boxes[(n*inRank+j)*2+1];
						boxes[(n*inRank+j)*2+1] = boxes[(i*inRank+j)*2+1];
						boxes[(i*inRank+j)*2+1] = temp;
					}
				}
				if(funcMin > boxesResult[n*3 + 2] ) {funcMin = boxesResult[n*3+2];boxMinIndex = n;}
			}

			if(funcMin < boxesResult[n*3]) break;
		}

		//std::cout << boxesResult[0] << "\t" << boxesResult[n*3+2] << "\t" << funcMin << "\n\n";

		curEps = std::abs(boxesResult[0] - funcMin);
		*outEps = curEps;
		*outMin = funcMin;
		memcpy(outBox,boxes + boxMinIndex*inRank*2,inRank*2*sizeof(double));
		std::cout << boxesResult[0] << "\t" << boxesResult[n*3] << "\t" << funcMin << "\t" << curEps << "\t" << n << "\n\n";
		if(curEps < inEps)
		{
			*status = 0;
			return;
		}
		numNewBoxes = n;

		tempRestBoxes = new double[numNewBoxes*inRank*2];
		memcpy(tempRestBoxes,boxes,numNewBoxes*inRank*2*sizeof(double));
		if(countIter > 0) delete [] restBoxes;
		restBoxes = tempRestBoxes;

		delete [] boxes;
		delete [] boxesResult;

		numRestBoxes = numNewBoxes;
		countIter++;
	}
	delete [] h;
}


__global__ void calculateLimitsOnCUDA(double *outLimits, double *inBox,int inRank,int inFunc, int numBoxes)
{
    int thread_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	if(thread_id < numBoxes){
		switch(inFunc){
		case 1:
				fnCalcFunLimitsMultiple2_CUDA(&inBox[thread_id*inRank*2], inRank, &outLimits[thread_id*3]);
				break;
		case 2:
				fnCalcFunLimitsHypebolic2_CUDA(&inBox[thread_id*inRank*2], inRank, &outLimits[thread_id*3]);
				break;
		case 3:
				fnCalcFunLimitsAluffiPentini2_CUDA(&inBox[thread_id*inRank*2], inRank, &outLimits[thread_id*3]);
				break;
		case 4:
				fnCalcFunLimitsRozenbroke_CUDA(&inBox[thread_id*inRank*2], inRank, &outLimits[thread_id*3]);
				break;
		}
	}
}


__global__ void calculateLimitsOnCUDA_deep(double *inBox,int inRank,int inFunc, int numBoxes, int *outWorkLen, double *outMins,double inFuncMin)
{
     int thread_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	int begin = thread_id*1024 * (2*inRank + 3);

	int deep = 2;
	int workLen = 1;
	double * work;
	double maxSide = 0;
	int maxRank = 0;
	int koeffSplit = 2;

	double funcMin = inFuncMin;
	double boxMin;
	double temp;
	double h;

	int i,j,k,s;

	double a,b;

	int maxDeep = 9;

	work = (double *) malloc(1024* (2*inRank + 3)*sizeof(double));
	for (i = 0; i < maxDeep; i++) 
	{
		for (j = 0; j < workLen; j++) 
		{
			maxSide = inBox[(begin+j*(2*inRank+3))+1] - inBox[(begin+j*(2*inRank+3))+1];
			maxRank = 0;
			for (k = 0; k < inRank; k++) 
			{
				if (inBox[(begin+j*(2*inRank+3))+2*k+1] - inBox[(begin+j*(2*inRank+3))+2*k] > maxSide) { maxSide = inBox[(begin+j*(2*inRank+3))+2*k+1] - inBox[(begin+j*(2*inRank+3))+2*k]; maxRank = k;}
			}

			h = maxSide / koeffSplit;
			for (s = 0; s < koeffSplit; s++) 
			{
				for (k = 0; k < inRank; k++) 
				{
					if (k == maxRank) 
					{
						work[((j*koeffSplit + s)*(2*inRank+3))+ 2*k] = inBox[(begin+j*(2*inRank+3))+2*k] + s*h;
						work[((j*koeffSplit + s)*(2*inRank+3))+ 2*k + 1] = inBox[(begin+j*(2*inRank+3))+2*k] + (s+1)*h;
					} 
					else 
					{
						work[((j*koeffSplit + s)*(2*inRank+3))+ 2*k] = inBox[(begin+j*(2*inRank+3))+2*k];
						work[((j*koeffSplit + s)*(2*inRank+3))+ 2*k + 1] = inBox[(begin+j*(2*inRank+3))+2*k+1];
					}
				}

				fnCalcFunLimitsRozenbroke_CUDA(&work[((j*koeffSplit + s)*(2*inRank+3))], inRank, &work[((j*koeffSplit + s)*(2*inRank+3))+ 2*inRank]);
			}	
		}

		workLen *= koeffSplit;
		boxMin = -1;

		for(j = 0; j < workLen; j++)
		{
			for(s = j + 1; s < workLen; s++)
			{
				if(work[j*(2*inRank+3) + 2*inRank] > work[s*(2*inRank+3) + 2*inRank]) 
				{
					for (int k = 0; k < inRank; k++) 
					{
						temp = work[j*(2*inRank+3) + 2*k];
						work[j*(2*inRank+3) + 2*k] = work[s*(2*inRank+3) + 2*k];
						work[s*(2*inRank+3) + 2*k] = temp;

						temp = work[j*(2*inRank+3) + 2*k + 1];
						work[j*(2*inRank+3) + 2*k + 1] = work[s*(2*inRank+3) + 2*k + 1];
						work[s*(2*inRank+3) + 2*k + 1] = temp;
					}

						temp = work[j*(2*inRank+3) + 2*inRank];
						work[j*(2*inRank+3) + 2*inRank] = work[s*(2*inRank+3) + 2*inRank];
						work[s*(2*inRank+3) + 2*inRank] = temp;

						temp = work[j*(2*inRank+3) + 2*inRank + 1];
						work[j*(2*inRank+3) + 2*inRank + 1] = work[s*(2*inRank+3) + 2*inRank + 1];
						work[s*(2*inRank+3) + 2*inRank + 1] = temp;

						temp = work[j*(2*inRank+3) + 2*inRank + 2];
						work[j*(2*inRank+3) + 2*inRank + 2] = work[s*(2*inRank+3) + 2*inRank + 2];
						work[s*(2*inRank+3) + 2*inRank + 2] = temp;
					}
					if(funcMin > work[s*(2*inRank+3) + 2*inRank + 2] ) {funcMin = work[s*(2*inRank+3) + 2*inRank + 2];boxMin = s;}
				}
				if(funcMin < work[j*(2*inRank+3) + 2*inRank] ) break;
			}
			
			memcpy(inBox + begin, work, workLen*(2*inRank + 3)*sizeof(double));
			workLen = j;
			if(workLen == 0 || workLen*koeffSplit >= 1024) break;
			
		}

		outWorkLen[thread_id] = workLen;
		outMins[thread_id*2] = funcMin;
		outMins[thread_id*2+1] = boxMin;
		free(work);
	
}



// Send Data to GPU to calculate limits
void sendDataToCuda(double *outLimits, const double *inBox, int inRank, int inFunc, int numBoxes)
{
    double *dev_outLimits = 0;
    double *dev_inBox = 0;

	int GridSize = ((numBoxes%BLOCK_SIZE == 0) ? numBoxes/BLOCK_SIZE : numBoxes/BLOCK_SIZE + 1);
	int numThreads = GridSize*BLOCK_SIZE;
	int sizeOutLimits = numThreads*3*sizeof(double);
	int sizeInBox = numThreads*inRank*2*sizeof(double);

	cudaEvent_t start, stop;

	CHECKED_CALL(cudaSetDevice(DEVICE));
    CHECKED_CALL(cudaMalloc((void **)&dev_outLimits, sizeOutLimits));
    CHECKED_CALL(cudaMalloc((void **)&dev_inBox, sizeInBox));
    CHECKED_CALL(cudaEventCreate(&start));
    CHECKED_CALL(cudaEventCreate(&stop));
	CHECKED_CALL(cudaMemcpy(dev_inBox, inBox, numBoxes*2*inRank*sizeof(double), cudaMemcpyHostToDevice));

	CHECKED_CALL(cudaEventRecord(start, 0));
	calculateLimitsOnCUDA<<<GridSize, BLOCK_SIZE>>>(dev_outLimits, dev_inBox, inRank, inFunc,numBoxes);
    CHECKED_CALL(cudaGetLastError());

    CHECKED_CALL(cudaEventRecord(stop, 0));
    CHECKED_CALL(cudaDeviceSynchronize());

    CHECKED_CALL(cudaMemcpy(outLimits, dev_outLimits, numBoxes*3*sizeof(double), cudaMemcpyDeviceToHost));

	float time;
    CHECKED_CALL(cudaEventElapsedTime(&time, start, stop));

    CHECKED_CALL(cudaEventDestroy(start));
    CHECKED_CALL(cudaEventDestroy(stop));
    CHECKED_CALL(cudaFree(dev_outLimits));
    CHECKED_CALL(cudaFree(dev_inBox));

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
    CHECKED_CALL(cudaMalloc((void **)&dev_inBox, sizeInBox));
	CHECKED_CALL(cudaMalloc((void **)&dev_workLen, numThreads*sizeof(int)));
	CHECKED_CALL(cudaMalloc((void **)&dev_mins, numThreads*sizeof(double)));
    CHECKED_CALL(cudaEventCreate(&start));
    CHECKED_CALL(cudaEventCreate(&stop));
	CHECKED_CALL(cudaMemcpy(dev_inBox, inBox, numBoxes*(2*inRank+3)*sizeof(double)*1024, cudaMemcpyHostToDevice));
	//CHECKED_CALL(cudaMemcpy(dev_workLen, workLen, numBoxes*sizeof(int), cudaMemcpyHostToDevice));

	CHECKED_CALL(cudaEventRecord(start, 0));
	globOptCUDA<<<GridSize, 1024>>>(dev_inBox, inRank,dev_workLen,dev_mins,funcMin, 0.001);	
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
    CHECKED_CALL(cudaFree(dev_workLen));
    CHECKED_CALL(cudaFree(dev_inBox));

}

void fnGetOptValueWithCUDA_deep(double *inBox, int inRank, int inNumBoxesSplitCoeff, double inEps, int inMaxIter, int inFunc, double *outBox, double*outMin, double *outEps,int *status)
{
	int numBoxes = 1024;
	double *boxes =  new double[numBoxes*(inRank*2+3)*1024];
	double h;
	int hInd;
	int numNewBoxes = 0;
	int *workLen;
	double *mins;

	int index = 0;
	int i,j,k,n;
	double temp;

	int countIter = 0;
	double curEps = inEps*10;
	
	double funcMin = 0;
	int boxMinIndex = 0;

	funcMin = 10;

	double indexMinLimit = 0;


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
	
	sendDataToCuda_deep(boxes, inRank, inFunc, numBoxes, workLen,mins,funcMin);

}


__const__ double rank = 10;


__global__ void globOptCUDA(double *inBox, int inRank, int *workLen, double *min, double inRec, double inEps)
{
	__shared__ double min_s[1024];
	__shared__ double workLen_s[1024];
	
	double minRec = inRec;
	int i, bInd, hInd;
	double curEps, h;
	
	int threadId = blockIdx.x * 1024 + threadIdx.x;
	
	workLen_s[threadId] = 1;
	min_s[threadId] = minRec;
	
	__syncthreads();
	
	
	while(workLen_s[threadId] > 0)
	{
		bInd = threadId*1024*(2*inRank+3) + (workLen_s[threadId] - 1)*(2*inRank+3);
		fnCalcFunLimitsRozenbroke_CUDA(inBox[threadId*1024*(2*inRank+3) + (workLen_s[threadId] - 1)*(2*inRank+3)], inRank);
		if(min_s[threadId] > inBox[bInd + 2*inRank + 2])
		{
			min_s[threadId] = inBox[bInd + 2*inRank + 2];
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
		curEps = min_s[threadId] - inBox[bInd + 2*inRank];
		curEps = curEps > 0 ? curEps : -curEps;	
		if(curEps < inEps)
		{
			--workLen_s[threadID];
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
}


#endif