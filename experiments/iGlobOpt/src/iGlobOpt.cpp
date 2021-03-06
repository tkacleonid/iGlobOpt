//============================================================================
// Name        : iGlobOpt.cpp
// Author      : Leonid Tkachenko
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================



#include "iGlobOpt.h"
#include "testFuncWithLib.h"
#include "testFuncWithOwnImpl.h"

void (*inFun)(const double *,int,double *);



int main() {
    int inDim = 2;

    double *inBox = NULL;
    double *argMin = NULL;
    double *outFunBounds = NULL;
    double outMin = 0.0;
    double inEps = 0.00001;
    int inNumBoxesSplitCoeff = 2;
    int status = -1;
    double initFunRecord = 0;
    bool isConfirm = true;


    int type_opt = 1;
    int type_test_fun = 1;

    printf("Введите тип функции оптимизации и номер тестовой функции через пробел: ");
    scanf("%d %d",&type_opt,&type_test_fun);
    printf("\nВведите точность оценивания: ");
    scanf("%lf",&inEps);
	inEps = 10;
for(int j = 0; j < 6; j++) {
	inEps /= 10;
	for (int k = 0; k < 1; k++) {
		switch (type_test_fun)
		{
			case 1:
				inFun = calcFunBoundsAckley1WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -32; inBox[1] = 32;
				inBox[2] = -32; inBox[3] = 32;
				

				break;
			case 2:
				inFun = calcFunBoundsAckley2WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -32; inBox[1] = 32;
				inBox[2] = -32; inBox[3] = 32;

				break;
			case 3:
				inFun = calcFunBoundsAckley3WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -32; inBox[1] = 32;
				inBox[2] = -32; inBox[3] = 32;

				break;
			case 4:
				inFun = calcFunBoundsAckley4WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -32; inBox[1] = 32;
				inBox[2] = -32; inBox[3] = 32;

				break;
			case 5:
				inFun = calcFunBoundsAdjimanWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -1; inBox[1] = 2;
				inBox[2] = -1; inBox[3] = 1;

				break;
			case 6:
				inFun = calcFunBoundsAlpine1WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -10; inBox[1] = 10;
				inBox[2] = -10; inBox[3] = 10;

				break;
			case 7:
				inFun = calcFunBoundsAlpine2WithLib;
				inDim = 3;
				inBox = new double[inDim * 2];

				inBox[0] = 0; inBox[1] = 10;
				inBox[2] = 0; inBox[3] = 10;
				inBox[4] = 0; inBox[5] = 10;

				break;
			case 8:
				inFun = calcFunBoundsBradWithLib;
				inDim = 3;
				inBox = new double[inDim * 2];

				inBox[0] = -0.25; inBox[1] = 0.25;
				inBox[2] = -0.01; inBox[3] = 2.5;
				inBox[4] = -0.01; inBox[5] = 2.5;

				break;
			case 9:
				inFun = calcFunBoundsBartelsConnWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -10; inBox[1] = 10;
				inBox[2] = -10; inBox[3] = 10;

				break;
			case 10:
				inFun = calcFunBoundsBealeWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -4.5; inBox[1] = 4.5;
				inBox[2] = -4.5; inBox[3] = 4.5;

				break;
			case 11:
				inFun = calcFunBoundsBiggsExpr2WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = 0; inBox[1] = 20;
				inBox[2] = 0; inBox[3] = 20;

				break;
			case 12:
				inFun = calcFunBoundsBiggsExpr3WithLib;
				inDim = 3;
				inBox = new double[inDim * 2];

				inBox[0] = 0; inBox[1] = 20;
				inBox[2] = 0; inBox[3] = 20;
				inBox[4] = 0; inBox[5] = 20;

				break;
			case 13:
				inFun = calcFunBoundsBiggsExpr4WithLib;
				inDim = 4;
				inBox = new double[inDim * 2];

				inBox[0] = 0; inBox[1] = 20;
				inBox[2] = 0; inBox[3] = 20;
				inBox[4] = 0; inBox[5] = 20;
				inBox[6] = 0; inBox[7] = 20;

				break;
			case 14:
				inFun = calcFunBoundsBiggsExpr5WithLib;
				inDim = 5;
				inBox = new double[inDim * 2];

				inBox[0] = 0; inBox[1] = 20;
				inBox[2] = 0; inBox[3] = 20;
				inBox[4] = 0; inBox[5] = 20;
				inBox[6] = 0; inBox[7] = 20;
				inBox[8] = 0; inBox[9] = 20;

				break;
			case 15:
				inFun = calcFunBoundsBiggsExpr6WithLib;
				inDim = 6;
				inBox = new double[inDim * 2];

				inBox[0] = -20; inBox[1] = 20;
				inBox[2] = -20; inBox[3] = 20;
				inBox[4] = -20; inBox[5] = 20;
				inBox[6] = -20; inBox[7] = 20;
				inBox[8] = -20; inBox[9] = 20;
				inBox[10] = -20; inBox[11] = 20;

				break;
			case 16:
				inFun = calcFunBoundsBirdWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -6.283185307179586476925286766559; inBox[1] = 6.283185307179586476925286766559;
				inBox[2] = -6.283185307179586476925286766559; inBox[3] = 6.283185307179586476925286766559;

				break;
			case 17:
				inFun = calcFunBoundsBohachevsky1WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -100; inBox[1] = 100;
				inBox[2] = -100; inBox[3] = 100;

				break;
			case 18:
				inFun = calcFunBoundsBohachevsky2WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -100; inBox[1] = 100;
				inBox[2] = -100; inBox[3] = 100;

				break;
			case 19:
				inFun = calcFunBoundsBoothWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -10; inBox[1] = 10;
				inBox[2] = -10; inBox[3] = 10;

				break;
			case 20:
				inFun = calcFunBoundsBohachevsky3WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -100; inBox[1] = 100;
				inBox[2] = -100; inBox[3] = 100;

				break;
			case 21:
				inFun = calcFunBoundsBoxBettsQuadraticSumWithLib;
				inDim = 3;
				inBox = new double[inDim * 2];

				inBox[0] = 0.9; inBox[1] = 1.2;
				inBox[2] = 9.0; inBox[3] = 11.2;
				inBox[4] = 0.9; inBox[5] = 1.2;

				break;
			case 22:
				inFun = calcFunBoundsBraninRCOSWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -5.0; inBox[1] = 10;
				inBox[2] = 0; inBox[3] = 15.0;

				break;
			case 23:
				inFun = calcFunBoundsBraninRCOS2WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -5.0; inBox[1] = 10;
				inBox[2] = 0; inBox[3] = 15.0;

				break;
			case 24:
				inFun = calcFunBoundsBrentWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -10.0; inBox[1] = 10.0;
				inBox[2] = -10.0; inBox[3] = 10.0;

				break;
			case 25:
				inFun = calcFunBoundsBrownWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -1.0; inBox[1] = 4.0;
				inBox[2] = -1.0; inBox[3] = 4.0;

				break;
			case 26:
				inFun = calcFunBoundsBukin2WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -15.0; inBox[1] = -5.0;
				inBox[2] = -3.0; inBox[3] = 3.0;

				break;
			case 27:
				inFun = calcFunBoundsBukin4WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -15.0; inBox[1] = -5.0;
				inBox[2] = -3.0; inBox[3] = 3.0;

				break;
			case 28:
				inFun = calcFunBoundsBukin6WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -15.0; inBox[1] = -5.0;
				inBox[2] = -3.0; inBox[3] = 3.0;

				break;
			case 29:
				inFun = calcFunBoundsCamelThreeHumpWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -5.0; inBox[1] = 5.0;
				inBox[2] = -5.0; inBox[3] = 5.0;

				break;
			case 30:
				inFun = calcFunBoundsCamelSixHumpWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -5.0; inBox[1] = 5.0;
				inBox[2] = -5.0; inBox[3] = 5.0;

				break;
			case 31:
				inFun = calcFunBoundsChichinadzeWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -30.0; inBox[1] = 30.0;
				inBox[2] = -30.0; inBox[3] = 30.0;

				break;
			case 32:
				inFun = calcFunBoundsChungReynoldsWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -100.0; inBox[1] = 100.0;
				inBox[2] = -100.0; inBox[3] = 100.0;

				break;
			case 33:
				inFun = calcFunBoundsColvilleWithLib;
				inDim = 4;
				inBox = new double[inDim * 2];

				inBox[0] = -10; inBox[1] = 10;
				inBox[2] = -10; inBox[3] = 10;
				inBox[4] = -10; inBox[5] = 10;
				inBox[6] = -10; inBox[7] = 10;

				break;
			case 34:
				inFun = calcFunBoundsComplexWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -2.0; inBox[1] = 2.0;
				inBox[2] = -2.0; inBox[3] = 2.0;

				break;
			case 35:
				inFun = calcFunBoundsCosineMixtureWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -1.0; inBox[1] = 1.0;
				inBox[2] = -1.0; inBox[3] = 1.0;

				break;
			case 36:
				inFun = calcFunBoundsCrossInTrayWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -15.0; inBox[1] = 15.0;
				inBox[2] = -15.0; inBox[3] = 15.0;

				break;
			case 37:
				inFun = calcFunBoundsCrossLegWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -10.0; inBox[1] = 10.0;
				inBox[2] = -10.0; inBox[3] = 10.0;

				break;
			case 38:
				inFun = calcFunBoundsCubeWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -10.0; inBox[1] = 10.0;
				inBox[2] = -10.0; inBox[3] = 10.0;

				break;
			case 39:
				inFun = calcFunBoundsDeb1WithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -1.0; inBox[1] = 1.0;
				inBox[2] = -1.0; inBox[3] = 1.0;

				break;

			case 40:
				inFun = calcFunBoundsDavisWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -100.0; inBox[1] = 100.0;
				inBox[2] = -100.0; inBox[3] = 100.0;

				break;
			case 41:
				inFun = calcFunBoundsDeckkersAartsWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -20.0; inBox[1] = 20.0;
				inBox[2] = -20.0; inBox[3] = 20.0;

				break;
			case 42:
				inFun = calcFunBoundsDixonPriceWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -10.0; inBox[1] = 10.0;
				inBox[2] = -10.0; inBox[3] = 10.0;

				break;
			case 43:
				inFun = calcFunBoundsDolanWithLib;
				inDim = 5;
				inBox = new double[inDim * 2];

				inBox[0] = -100.0; inBox[1] = 100.0;
				inBox[2] = -100.0; inBox[3] = 100.0;
				inBox[4] = -100.0; inBox[5] = 100.0;
				inBox[6] = -100.0; inBox[7] = 100.0;
				inBox[8] = -100.0; inBox[9] = 100.0;

				break;
			case 44:
				inFun = calcFunBoundsDropWaveWithLib;
				inDim = 2;
				inBox = new double[inDim * 2];

				inBox[0] = -5.12; inBox[1] = 5.12;
				inBox[2] = -5.12; inBox[3] = 5.12;

				break;
			case 45:
				 inFun = calcFunBoundsRosenbrockWithLib5;
				 inDim = 5;
				 inBox = new double[inDim * 2];
				 for(int i = 0; i < inDim; i++)
				 {
					 inBox[i*2] = -30.0;
					 inBox[i*2+1] = 30.0;
				 }
				 break;
			case 46:
				 inFun = calcFunBoundsRosenbrockWithLib10;
				 inDim = 10;
				 inBox = new double[inDim * 2];
				 for(int i = 0; i < inDim; i++)
				 {
					 inBox[i*2] = -30.0;
					 inBox[i*2+1] = 30.0;
				 }
				 break;
			case 47:
				 inFun = calcFunBoundsRosenbrockWithLib15;
				 inDim = 15;
				 inBox = new double[inDim * 2];
				 for(int i = 0; i < inDim; i++)
				 {
					 inBox[i*2] = -30.0;
					 inBox[i*2+1] = 30.0;
				 }

				 break;
			case 48:
				 inFun = calcFunBoundsRosenbrockWithLib20;
				 inDim = 20;
				 inBox = new double[inDim * 2];
				 for(int i = 0; i < inDim; i++)
				 {
					 inBox[i*2] = -30.0;
					 inBox[i*2+1] = 30.0;
				 }
				 break;
			case 49:
				 inFun = calcFunBoundsRosenbrockModifiedWithLib;
				 inDim = 2;
				 inBox = new double[inDim * 2];


				 inBox[0] = -2.0; inBox[1] = 2.0;
				 inBox[2] = -2.0; inBox[3] = 2.0;

				 break;
			case 50:
				 inFun = fnCalcFunLimitsRozenbroke;
				 inDim = 5;
				 inBox = new double[inDim * 2];
				 for(int i = 0; i < inDim; i++)
				 {
					 inBox[i*2] = -30.0;
					 inBox[i*2+1] = 30.0;
				 }
				 break;
			case 51:
				 inFun = fnCalcFunLimitsRozenbroke;
				 inDim = 10;
				 inBox = new double[inDim * 2];
				 for(int i = 0; i < inDim; i++)
				 {
					 inBox[i*2] = -30.0;
					 inBox[i*2+1] = 30.0;
				 }
				 break;
			case 52:
				 inFun = fnCalcFunLimitsRozenbroke;
				 inDim = 15;
				 inBox = new double[inDim * 2];
				 for(int i = 0; i < inDim; i++)
				 {
					 inBox[i*2] = -30.0;
					 inBox[i*2+1] = 30.0;
				 }
				 break;
			case 53:
				 inFun = fnCalcFunLimitsRozenbroke;
				 inDim = 20;
				 inBox = new double[inDim * 2];
				 for(int i = 0; i < inDim; i++)
				 {
					 inBox[i*2] = -30.0;
					 inBox[i*2+1] = 30.0;
				 }
				 break;
			case 54:
				 inFun = fnCalcFunLimitsAluffiPentini2;
				 inDim = 2;
				 inBox = new double[inDim * 2];
				 for(int i = 0; i < inDim; i++)
				 {
					 inBox[i*2] = -30.0;        
					 inBox[i*2+1] = 30.0;    
				 }
				 break;
		case 55:
				 inFun = fnCalcFunLimitsStyblinski;
				 inDim = 3;
				 inBox = new double[inDim * 2];
				 for(int i = 0; i < inDim; i++)
				 {
					 inBox[i*2] = -5.0;
					 inBox[i*2+1] = 5.0;
				 }
				 initFunRecord = 39.1661657038*inDim;
				 break;			 

			default: exit(0);
		}

    argMin = new double[inDim * 2];
	if(!isConfirm) {
		inFun(inBox,inDim,outFunBounds);
		initFunRecord = outFunBounds[GO_POSITION_FUN_RECORD];
	}
	

    auto start = std::chrono::high_resolution_clock::now();

   	GlobOptErrors st;
    switch (type_opt) {
        case 1: calcOptValueOnCPUBFS(inBox,1, inDim, inNumBoxesSplitCoeff, inFun, inEps, &outMin, &st,argMin); break;
        case 2: calcOptValueOnCPUBFSWithMmap(inBox, 1,inDim, inNumBoxesSplitCoeff,   inFun, inEps, &outMin,  &st,argMin); break;
        case 3: calcOptValueOnCPUBFSWithOMP(inBox, 1,inDim, inNumBoxesSplitCoeff,  inFun, inEps, &outMin,  &st,argMin); break;
        case 4: calcOptValueOnCPUBFSWithMmapAndOMP(inBox, 1,inDim, inNumBoxesSplitCoeff,  inFun, inEps, &outMin,  &st,argMin, initFunRecord); break;
        default: calcOptValueOnCPUBFS(inBox,1, inDim, inNumBoxesSplitCoeff,  inFun, inEps, &outMin,  &st,argMin);
    }

	
	
	
    auto end = std::chrono::high_resolution_clock::now();
	

	if(st == GO_WORKBUFFER_IS_FULL)
	{
		std::cout << "Optimization is not finished: Work buffer is full. Reached function record " << outMin << std::endl;
	}
	else	if(st == GO_WORKBUFFER_IS_EMPTY)
	{
		std::cout << "Optimization is not finished: Work buffer is empty. Reached function record " << outMin << std::endl;
	}
	
	
	else if(st == GO_SUCCESS)
	{
	    std::cout << "Result: ";
	    for (int i = 0; i < inDim; i++) {
	        printf("[%.8lf; %.8lf]\t",argMin[i*2],argMin[i*2+1]);
	    }

	    std::cout << "\n";
	    printf("Record of function: [%.8lf]\t",outMin);
	    std::cout << "\n";
	    std::cout << "time in millisecs: " << ((std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count())/1 << "\t";
	    std::cout << "\n\n";

	}
	
}


std::cout << "\n\n";
	}
    delete [] inBox;
    delete [] argMin;

    return 0;
}
