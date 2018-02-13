
#include "CUDAGlobalOptimization.h"
#include "interval.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <exception>



int main()
{
	
    int inRank = 2;

	double *inBox = new double[inRank*2];
	double *outBox = new double[inRank*2];
	double outMin = 0.0;
	double inEps = 0.001;
	int status = -1;


	for(int i = 0; i < inRank; i++)
	{
		inBox[i*2] = -50.0;
		inBox[i*2+1] = 50.0;
	}

	std::cout << "start Main\n";

	std::cout << "start \n";

	fnGetOptValueWithCUDA(inBox, inRank, inEps, outBox, &outMin, &status);

	std::cout << "min = " << outMin << "\t";

	//delete [] inBox;
	//delete [] outBox;

	return 0;
}




