
#include "CUDAGlobalOptimization.h"
//#include "CUDAGlobalOptimization.cpp"
#include "interval.h"
#include <time.h>
#include <stdio.h>
#include <iostream>

const int NUM_BLOCKS = 1;
const int BLOCK_SIZE = 512;
const int NUM_GRIDS = 1;
const int MAX_BOXES = 1024;

void fnGetOptValueWithCUDA(double *inBox, int inRank, int inBlockSize, int MaxGridSize, int inNumBoxesSplitCoeff, double inEps, int inMaxIter, int inFunc, double *outBox, double*outMin, double *outEps,int *status);

__global__ void calculateLimitsOnCUDA2(){int thread_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;}

__global__ void calculateLimitsOnCUDA(double *numBoxes, const double *inBox,int inRank)
{
    int thread_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;   

	fnCalcFunLimitsRozenbroke(&inBox[thread_id*inRank*2], inRank, &outLimits[thread_id*3]);

}

int main()
{
    int inRank = 4;
	time_t t_start, t_end;

	double *inBox = new double[inRank*2];
	double *outBox = new double[inRank*3];
	double outMin = 0.0;
	double inEps = 0.0001;
	double outEps = 0.0;
	double inMaxIter = 100;
	int inNumBoxesSplitCoeff = GRID_SIZE;
	int status = -1;

	for(int i = 0; i < inRank; i++)
	{
		inBox[i*2] = -30.0;
		inBox[i*2+1] = 30.0;
	}


	for(int i = 0; i < 1; i++)
	{
		fnGetOptValueWithCUDA(inBox, inRank, BLOCK_SIZE, 2 << 30, inNumBoxesSplitCoeff, inEps, inMaxIter, 4, outBox,&outMin, &outEps, &status);
	}



	delete [] inBox;
	delete [] outBox;

	return 0;
}

// Send Data to GPU to calculate limits
void sendDataToCuda(double *outLimits, const double *inBox, int inRank, int inGridSize,int inBlockSize, int devID, int inFunc)
{
    double *dev_outLimits = 0;
    double *dev_inBox = 0;
	int numThreads = inGridSize*inBlockSize;
	int sizeOutLimits = numThreads*3*sizeof(double);
	int sizeInBox = numThreads*inRank*2*sizeof(double);

	cudaEvent_t start, stop;

	CHECKED_CALL(cudaSetDevice(devID));
    CHECKED_CALL(cudaMalloc((void **)&dev_outLimits, sizeOutLimits));
    CHECKED_CALL(cudaMalloc((void **)&dev_inBox, sizeInBox));
    CHECKED_CALL(cudaEventCreate(&start));
    CHECKED_CALL(cudaEventCreate(&stop));
	CHECKED_CALL(cudaMemcpy(dev_inBox, inBox, sizeInBox, cudaMemcpyHostToDevice));

	CHECKED_CALL(cudaEventRecord(start, 0));

	calculateLimitsOnCUDA<<<inGridSize, inBlockSize>>>(dev_outLimits, dev_inBox, inRank, inFunc);
    CHECKED_CALL(cudaGetLastError());

    CHECKED_CALL(cudaEventRecord(stop, 0));
    CHECKED_CALL(cudaDeviceSynchronize());

    CHECKED_CALL(cudaMemcpy(outLimits, dev_outLimits, sizeOutLimits, cudaMemcpyDeviceToHost));

	float time;
    CHECKED_CALL(cudaEventElapsedTime(&time, start, stop));

    CHECKED_CALL(cudaEventDestroy(start));
    CHECKED_CALL(cudaEventDestroy(stop));
    CHECKED_CALL(cudaFree(dev_outLimits));
    CHECKED_CALL(cudaFree(dev_inBox));

	std::cout << time << "\n";

}

void fnGetOptValueWithCUDA(double *inBox, int inRank, int inBlockSize, int MaxGridSize, int inNumBoxesSplitCoeff, double inEps, int inMaxIter, int inFunc, double *outBox, double*outMin, double *outEps,int *status)
{
	//int numThreads = inBlockSize*GridSize;
	//int splitCoeff = pow((double)numThreads,1.0/inRank);
	int numBoxes = BLOCK_SIZE*NUM_BLOCKS*NUM_GRIDS;
	double *boxes =  new double[numBoxes*inRank*2];
	double *boxesResult = new double[numBoxes*3];
	double *restBoxes = new double[inRank*2];
	double *newRestBoxes = NULL,*tempRestBoxes = NULL;
	double h;
	int numNewBoxes = 0;
	int numCudaRuns = GridSize/MaxGridSize;

	memcpy(restBoxes,inBox,inRank*2*sizeof(double));

	int numRestBoxes = 1;
	int countBox = 0;
	int index = 0;
	int i,j,k,n;
	double temp;

	int countIter = 0;
	double curEps = inEps*10;
	
	double funcMin = 0;
	int boxMinIndex = 0;

	boxes = new double[numRestBoxes*numBoxes*inRank*5];
	k = 0;
	h = (restBoxes[(k*inRank+0)*2 + 1] - restBoxes[(k*inRank+0)*2])/numBoxes;
	for(n = 0; n < numBoxes; n++)
	{
		for(i = 0; i < inRank; i++)
		{
			
			boxes[(n*inRank+i)*5] = restBoxes[(k*inRank+i)*2] + h[i]*index;
			boxes[(n*inRank+i)*5 + 1] = restBoxes[(k*inRank+i)*2] + h[i]*(index+1);
		}
	
	}


	int *dev_num_boxes = 0;
	double *dev_outLimits = 0;
    double *dev_inBox = 0;
	int numThreads = numBoxes;
	//int sizeOutLimits = numThreads*3*sizeof(double);
	int sizeInBox = numThreads*inRank*5*sizeof(double);
	int sizeNumBoxes = numThreads*sizeof(int);
	
	int *host_num_boxes = new int[numThreads];
	
	for(int i  =0; i < sizeNumBoxes; i++)
	{
		host_num_boxes[i] = 0;
	}

	cudaEvent_t start, stop;

	CHECKED_CALL(cudaSetDevice(devID));
    //CHECKED_CALL(cudaMalloc((void **)&dev_outLimits, sizeOutLimits));
    CHECKED_CALL(cudaMalloc((void **)&dev_inBox, sizeInBox));
	CHECKED_CALL(cudaMalloc((void **)&dev_num_boxes, sizeNumBoxes));
    CHECKED_CALL(cudaEventCreate(&start));
    CHECKED_CALL(cudaEventCreate(&stop));
	CHECKED_CALL(cudaMemcpy(dev_inBox, boxes, sizeInBox, cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_num_boxes, host_num_boxes, sizeNumBoxes, cudaMemcpyHostToDevice));

	CHECKED_CALL(cudaEventRecord(start, 0));

	calculateLimitsOnCUDA<<<NUM_BLOCKS, BLOCK_SIZEe>>>(dev_num_boxes, dev_inBox, inRank);
    
	CHECKED_CALL(cudaGetLastError());

    CHECKED_CALL(cudaEventRecord(stop, 0));
    CHECKED_CALL(cudaDeviceSynchronize());

    CHECKED_CALL(cudaMemcpy(host_num_boxes, dev_num_boxes, sizeNumBoxes, cudaMemcpyDeviceToHost));

	float time;
    CHECKED_CALL(cudaEventElapsedTime(&time, start, stop));

    CHECKED_CALL(cudaEventDestroy(start));
    CHECKED_CALL(cudaEventDestroy(stop));
    CHECKED_CALL(cudaFree(dev_num_boxes));
    CHECKED_CALL(cudaFree(dev_inBox));

	std::cout << time << "\n";
	


}

