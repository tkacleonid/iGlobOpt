
#include "CUDAGlobalOptimization.h"
//#include "CUDAGlobalOptimization.cpp"
#include "interval.h"
#include <time.h>
#include <stdio.h>
#include <iostream>


void fnGetOptValueWithCUDA(double *inBox, int inRank, int inBlockSize, int MaxGridSize, int inNumBoxesSplitCoeff, double inEps, int inMaxIter, int inFunc, double *outBox, double*outMin, double *outEps,int *status);

__global__ void calculateLimitsOnCUDA2(){int thread_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;}

__global__ void calculateLimitsOnCUDA(double *outLimits, const double *inBox,int inRank,int inFunc)
{
    int thread_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
	switch(inFunc){
	case 1:
			fnCalcFunLimitsMultiple2(&inBox[thread_id*inRank*2], inRank, &outLimits[thread_id*3]);
			break;
	case 2:
			fnCalcFunLimitsHypebolic2(&inBox[thread_id*inRank*2], inRank, &outLimits[thread_id*3]);
			break;
	case 3:
			fnCalcFunLimitsAluffiPentini2(&inBox[thread_id*inRank*2], inRank, &outLimits[thread_id*3]);
			break;
	case 4:
			fnCalcFunLimitsRozenbroke(&inBox[thread_id*inRank*2], inRank, &outLimits[thread_id*3]);
			break;

	}
}

int main()
{
    int inRank = 4;
	time_t t_start, t_end;

	double *inBox = new double[inRank*2];
	double *outBox = new double[inRank*2];
	double outMin = 0.0;
	double inEps = 0.0001;
	double outEps = 0.0;
	double inMaxIter = 100;
	int inNumBoxesSplitCoeff = GRID_SIZE;
	int status = -1;

	for(int i = 0; i < inRank; i++)
	{
		inBox[i*2] = -20.0;
		inBox[i*2+1] = 20.0;
	}

	time(&t_start);
	for(int i = 0; i < 1; i++)
	{
		fnGetOptValueWithCUDA(inBox, inRank, BLOCK_SIZE, 2 << 30, inNumBoxesSplitCoeff, inEps, inMaxIter, 4, outBox,&outMin, &outEps, &status);
	}

	time(&t_end);
	std::cout << "Result: ";
	for(int i = 0; i < inRank; i++)
	{
		std::cout << "[" << outBox[i*2] << "; " << outBox[i*2 + 1]  << "]\t";
	}

	std::cout << "\n";
	std::cout << "min = " << outMin << "\t";
	std::cout << "eps = " << outEps;
	std::cout << "\n";
	std::cout << "time = " << (t_end-t_start) << "seconds";
	std::cout << "\n";

	std::cin.get();

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
	int numBoxes = pow((double) inNumBoxesSplitCoeff,inRank);
	int GridSize = numBoxes / inBlockSize;
	double *boxes =  new double[numBoxes*inRank*2];
	double *boxesResult = new double[numBoxes*3];
	double *restBoxes = new double[inRank*2];
	double *newRestBoxes = NULL,*tempRestBoxes = NULL;
	double *h = new double[inRank];
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

	while((countIter < inMaxIter) && (curEps >= inEps))
	{
		boxes = new double[numRestBoxes*numBoxes*inRank*2];
		boxesResult = new double[numRestBoxes*numBoxes*3];
		for(k = 0; k < numRestBoxes; k++)
		{
			for(i = 0; i < inRank; i++)
			{
				h[i] = (restBoxes[(k*inRank+i)*2 + 1] - restBoxes[(k*inRank+i)*2])/inNumBoxesSplitCoeff;
			}
			for(n = 0; n < numBoxes; n++)
			{
				for(i = 0; i < inRank; i++)
				{
					index = ((n % numBoxes) % (long) pow((double)inNumBoxesSplitCoeff,i+1))/((long)pow((double)inNumBoxesSplitCoeff,i));
					boxes[((k*numBoxes + n)*inRank+i)*2] = restBoxes[(k*inRank+i)*2] + h[i]*index;
					boxes[((k*numBoxes + n)*inRank+i)*2 + 1] = restBoxes[(k*inRank+i)*2] + h[i]*(index+1);
				}
				//inFun(&boxes[((k*numBoxes + n)*inRank)*2],inRank,&boxesResult[(k*numBoxes + n)*3]);
			}
		}

		GridSize = (numRestBoxes*numBoxes) / inBlockSize;

		sendDataToCuda(boxesResult, boxes, inRank, GridSize,inBlockSize, DEVICE, inFunc);



		funcMin = boxesResult[2];
		boxMinIndex = 0;
		for(n = 0; n < numRestBoxes*numBoxes; n++)
		{
			//std::cout << boxesResult[n*3] << "\n";
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
			}
			if(funcMin > boxesResult[n*3 + 2] ) 
			{
					funcMin = boxesResult[n*3+2];
					boxMinIndex = n;
			}
		}

		n = 0;
		//funcMin = boxesResult[2];
		while(n < numRestBoxes*numBoxes)
		{
			curEps = std::abs(boxesResult[n*3] - funcMin);
			//std::cout << curEps << " " << funcMin << "\n";
			if(curEps < inEps)
			{
				*outEps = curEps;
				*outMin = funcMin;
				memcpy(outBox,boxes + boxMinIndex*inRank*2,inRank*2*sizeof(double));
				*status = 0;
				return;
			}

			

			if(boxesResult[n*3] > funcMin) break;		
			n++;
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

