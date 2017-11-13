
#include "CUDAGlobalOptimization.h"
#include "interval.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <exception>



int main()
{
	
	
	
    int inRank = 4;

	double *inBox = new double[inRank*2];
	double *outBox = new double[inRank*2];
	double outMin = 0.0;
	double inEps = 0.1;
	double outEps = 0.0;
	double inMaxIter = 100;
	int status = -1;


	inRank = 4;
	for(int i = 0; i < inRank; i++)
	{
		inBox[i*2] = -30.0;
		inBox[i*2+1] = 30.0;
	}

	//long long limitHeapSize = 1024*20000*sizeof(double)*(2*inRank + 3);

	
	CHECKED_CALL(cudaSetDevice(DEVICE));
	
	std::cout << "start Main\n";
	
	//CHECKED_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,limitHeapSize));


	std::cout << "start \n";
	//fnGetOptValueOnCPU(inBox, inRank, 2, inEps, inMaxIter, fnCalcFunLimitsAluffiPentini2, outBox,&outMin, &outEps, &status);
	fnGetOptValueWithCUDA_deep(inBox, inRank, 16, inEps, inMaxIter, 4, outBox,&outMin, &outEps, &status);

	std::cout << "min = " << outMin << "\t";

	//delete [] inBox;
	//delete [] outBox;

	return 0;
}




