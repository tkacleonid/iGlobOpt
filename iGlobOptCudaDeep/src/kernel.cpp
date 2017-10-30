
#include "CUDAGlobalOptimization.h"
#include "CPUGlobalOptimization.h"
#include "interval.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <exception>


void testFunc(double *inBox, int inRank, int inNumBoxesSplitCoeffCPU, int inNumBoxesSplitCoeffGPU, double inEps, int inMaxIter, int inFun, double * outBox, double * outMin, double *outEps, int *status, std::ofstream &fout){

	time_t t_startCPU, t_endCPU,t_startGPU, t_endGPU;
	try{
		switch(inFun){
		case 1:
			fout << "Произведение параметров (" << inRank << " переменных)" << "\t";
			time(&t_startCPU);
			fnGetOptValueOnCPU(inBox, inRank, inNumBoxesSplitCoeffCPU, inEps, inMaxIter, fnCalcFunLimitsMultiple2, outBox,outMin, outEps, status);
			time(&t_endCPU);
			
			time(&t_startGPU);
			fnGetOptValueWithCUDA(inBox, inRank, inNumBoxesSplitCoeffGPU, inEps, inMaxIter, 1, outBox,outMin, outEps, status);
			time(&t_endGPU);
			break;
		case 2:
			fout << "Гиперболическая функция (" << inRank << " переменных)" << "\t";
			time(&t_startCPU);
			fnGetOptValueOnCPU(inBox, inRank, inNumBoxesSplitCoeffCPU, inEps, inMaxIter, fnCalcFunLimitsHypebolic2, outBox,outMin, outEps, status);
			time(&t_endCPU);
			
			time(&t_startGPU);
			fnGetOptValueWithCUDA(inBox, inRank, inNumBoxesSplitCoeffGPU, inEps, inMaxIter, 2, outBox,outMin, outEps, status);
			time(&t_endGPU);
			break;
		case 3:
			fout << "Функция Алуффи-Пентини (" << inRank << " переменных)" << "\t";
			time(&t_startCPU);
			fnGetOptValueOnCPU(inBox, inRank, inNumBoxesSplitCoeffCPU, inEps, inMaxIter, fnCalcFunLimitsAluffiPentini2, outBox,outMin, outEps, status);
			time(&t_endCPU);
			
			time(&t_startGPU);
			fnGetOptValueWithCUDA(inBox, inRank, inNumBoxesSplitCoeffGPU, inEps, inMaxIter, 3, outBox,outMin, outEps, status);
			time(&t_endGPU);
			break;
		case 4:
			fout << "Функция розенброка (" << inRank << " переменных)" << "\t";
			time(&t_startCPU);
			fnGetOptValueOnCPU(inBox, inRank, inNumBoxesSplitCoeffCPU, inEps, inMaxIter, fnCalcFunLimitsRozenbroke, outBox,outMin, outEps, status);
			time(&t_endCPU);
			
			time(&t_startGPU);
			fnGetOptValueWithCUDA(inBox, inRank, inNumBoxesSplitCoeffGPU, inEps, inMaxIter, 4, outBox,outMin, outEps, status);
			time(&t_endGPU);
			break;
		}
	}
	catch(std::exception ex){
		std::cout << "Произошла ошибка при решении задачи оптимизации: " << ex.what() << "\n";
		exit(EXIT_FAILURE); 
	}
	fout << inEps << "\t";
	fout << inNumBoxesSplitCoeffCPU << "\t";
	fout << inNumBoxesSplitCoeffGPU << "\t";
	fout << (t_endCPU-t_startCPU) << "\t";
	fout << (t_endGPU-t_startGPU) << "\n";
}

int main()
{
    int inRank = 2;

	double *inBox = new double[inRank*2];
	double *outBox = new double[inRank*2];
	double outMin = 0.0;
	double inEps = 0.1;
	double outEps = 0.0;
	double inMaxIter = 100;
	int inNumBoxesSplitCoeff = 2;
	int status = -1;
	int numTestCycles = 10;


	inRank = 2;
	for(int i = 0; i < inRank; i++)
	{
		inBox[i*2] = -30.0;
		inBox[i*2+1] = 30.0;
	}

	long long limitHeapSize = 1024*20000*sizeof(double)*(2*inRank + 3);

	CHECKED_CALL(cudaSetDevice(DEVICE));
	CHECKED_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,limitHeapSize));


	//fnGetOptValueOnCPU(inBox, inRank, 2, inEps, inMaxIter, fnCalcFunLimitsAluffiPentini2, outBox,&outMin, &outEps, &status);
	fnGetOptValueWithCUDA_deep(inBox, inRank, 16, inEps, inMaxIter, 4, outBox,&outMin, &outEps, &status);

	std::cout << "min = " << outMin << "\t";

	delete [] inBox;
	delete [] outBox;

	return 0;
}




