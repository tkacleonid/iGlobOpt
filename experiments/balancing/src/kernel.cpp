
#include "balancing.hpp"

int main()
{	
	int dim = 2;
	int numThreads = 100;
	int maxBoxesPerThread = 100000;


	double *boxes = new double[(2*dim+3) * numThreads*maxBoxesPerThread];
	int *workLen = new int[numThreads];
	
	double *tempBoxes = new double[(2*dim+3) * numThreads*maxBoxesPerThread];
	int *tempWorkLen = new int[numThreads];
	
	BalancingInfo balancingInfo;
	
	printf("..........................\n");
	printf("Stage 1: \n");
	printf("\tDimension: %d\n",dim);
	printf("\tNumber of threads: %d\n",numThreads);
	printf("\tMax number of boxes for thread: %d\n",maxBoxesPerThread);
	
	
	
	printf("Initializing boxes\n");
	initializeBoxes(boxes, workLen, numThreads, maxBoxesPerThread, dim);
	
	memcpy(tempBoxes,boxes,sizeof(double)*(2*dim+3) * numThreads*maxBoxesPerThread);
	memcpy(tempWorkLen,workLen,sizeof(int)*numThreads);	
	printf("Testing balancing on CPU (version 1)\n");
	balancingInfo = balancingOnCPU_v2(tempBoxes, tempWorkLen, numThreads, maxBoxesPerThread, dim);
	printf("numberOfMemoryCopies = %d\n",balancingInfo.numberOfMemoryCopies);
	printf("time = %d\n",balancingInfo.time);
	printf("numAllBoxes = %d\n",balancingInfo.numAllBoxes);
	printf("numAverageBoxes = %d\n",balancingInfo.numAverageBoxes);
	
	memcpy(tempBoxes,boxes,sizeof(double)*(2*dim+3) * numThreads*maxBoxesPerThread);
	memcpy(tempWorkLen,workLen,sizeof(int)*numThreads);	
	printf("\nTesting balancing on CPU (version 2)\n");
	balancingInfo = balancingOnCPU_v3(tempBoxes, tempWorkLen, numThreads, maxBoxesPerThread, dim);
	printf("numberOfMemoryCopies = %d\n",balancingInfo.numberOfMemoryCopies);
	printf("time = %d\n",balancingInfo.time);
	printf("numAllBoxes = %d\n",balancingInfo.numAllBoxes);
	printf("numAverageBoxes = %d\n",balancingInfo.numAverageBoxes);
	
	
	delete [] boxes;
	delete [] workLen;
	delete [] tempBoxes;
	delete [] tempWorkLen;
	
	printf("..........................\n");
		
	return 0;
}
