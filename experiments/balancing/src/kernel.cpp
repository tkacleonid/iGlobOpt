
#include "balancing.hpp"

int main()
{	
	int dim = 2;
	int numThreads = 1000;
	int maxBoxesPerThread = 1000;


	double *boxes = new double[(2*dim+3) * numThreads*maxBoxesPerThread];
	int *workLen = new int[numThreads];
	BalancingInfo balancingInfo;
	
	printf("..........................\n");
	printf("Stage 1: \n");
	printf("\tDimension: %d\n",dim);
	printf("\tNumber of threads: %d\n",numThreads);
	printf("\tMax number of boxes for thread: %d\n",maxBoxesPerThread);
	
	
	
	printf("Initializing boxes\n");
	initializeBoxes(boxes, workLen, numThreads, maxBoxesPerThread, dim);
	printf("Testing balancing on CPU (version 1)\n");
	balancingInfo = balancingOnCPU_v3(boxes, workLen, numThreads, maxBoxesPerThread, dim);
	printf("numberOfMemoryCopies = %d\n",balancingInfo.numberOfMemoryCopies);
	printf("time = %d\n",balancingInfo.time);
	
	
	delete [] boxes;
	delete [] workLen;
	
	printf("..........................\n");
		
	return 0;
}
