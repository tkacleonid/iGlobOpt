
#include "CUDAGlobalOptimization.h"
#include "interval.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <exception>



int main()
{
	
    int inDim = 3;

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

	for (int i = 0; i < 5; i++) {
		inEps /= 10;
		fnGetOptValueWithCUDA_v2(inBox, inDim, inEps, outBox, &outMin, &status, funRecord, "testGPU21");
		std::cout << "min = " << outMin << "\n\n";
	}

	delete [] inBox;
	delete [] outBox;

	return 0;
}




