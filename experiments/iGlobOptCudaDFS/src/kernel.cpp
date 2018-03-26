
#include "CUDAGlobalOptimization.h"
#include "interval.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <exception>



int main()
{
	
    int inDim = 2;

	double *inBox = new double[inDim*2];
	double *outBox = new double[inDim*2];
	double outMin = 0.0;
	double inEps = 1;
	int status = -1;
	double funRecord = -39.1661657038*inDim;

	for(int i = 0; i < inDim; i++) {
		inBox[i*2] = -5.0;
		inBox[i*2+1] = 5.0;
	}

	std::cout << "start Main\n";
	std::cout << "start \n";

	for (int i = 0; i < 7; i++) {
		inEps /= 10;
		fnGetOptValueWithCUDA(inBox, inDim, inEps, outBox, &outMin, &status, funRecord, "testGPU1");
		std::cout << "min = " << outMin << "\n\n";
	}

	delete [] inBox;
	delete [] outBox;

	return 0;
}




