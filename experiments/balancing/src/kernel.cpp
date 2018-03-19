
#include "balancing.hpp"

int main()
{	
	int dim = 2;
	int numThreads = 1024;
	int maxBoxesPerThread = 1000;


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
	
	printf("\n\n");
	for (int i = 0; i < numThreads; i++) {
		printf("%d\t", tempWorkLen[i]);
	}
	printf("\n\n");
	
	
	memcpy(tempBoxes,boxes,sizeof(double)*(2*dim+3) * numThreads*maxBoxesPerThread);
	memcpy(tempWorkLen,workLen,sizeof(int)*numThreads);	
	printf("\nTesting balancing on CPU (version 2)\n");
	balancingInfo = balancingOnCPU_v3(tempBoxes, tempWorkLen, numThreads, maxBoxesPerThread, dim);
	printf("numberOfMemoryCopies = %d\n",balancingInfo.numberOfMemoryCopies);
	printf("time = %d\n",balancingInfo.time);
	printf("numAllBoxes = %d\n",balancingInfo.numAllBoxes);
	printf("numAverageBoxes = %d\n",balancingInfo.numAverageBoxes);

/*	
	for (int i = 0; i < numThreads; i++) {
		printf("%d\t", workLen[i]);
	}
	printf("\n\n");
*/
	
	
	memcpy(tempBoxes,boxes,sizeof(double)*(2*dim+3) * numThreads*maxBoxesPerThread);
	memcpy(tempWorkLen,workLen,sizeof(int)*numThreads);	
/*	
	for (int i = 0; i < numThreads; i++) {
		printf("%d\t", tempWorkLen[i]);
	}
	printf("\n\n");
*/	
	
	printf("\nTesting balancing on GPU (version 1)\n");
	balancingInfo = balancingOnGPU_v1(tempBoxes, tempWorkLen, numThreads, maxBoxesPerThread, dim);
	printf("numberOfMemoryCopies = %d\n",balancingInfo.numberOfMemoryCopies);
	printf("time = %f\n",balancingInfo.time);
	printf("numAllBoxes = %d\n",balancingInfo.numAllBoxes);
	printf("numAverageBoxes = %d\n",balancingInfo.numAverageBoxes);
	
	printf("\n\n");
	for (int i = 0; i < numThreads; i++) {
		printf("%d\t", tempWorkLen[i]);
	}
	printf("\n\n");
	
	
	delete [] boxes;
	delete [] workLen;
	delete [] tempBoxes;
	delete [] tempWorkLen;
	
	printf("..........................\n");
		
	return 0;
}
