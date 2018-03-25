
#include "balancing.hpp"

int main()
{	
	int dim = 2;
	int numThreads = BLOCK_SIZE;
	int maxBoxesPerThread = 8;	

	double *boxes;
	int *workLen;
	
	double *tempBoxes;
	int *tempWorkLen;
	
	BalancingInfo balancingInfo;
	
	std::ofstream outfile;
	outfile.open("balancingTime",std::ios_base::app);
	if (outfile.fail())
			throw std::ios_base::failure(std::strerror(errno));
	for (int i = 2; i < 16384; i*=2) {
		maxBoxesPerThread = i;
		
		boxes = new double[(2*dim+3) * numThreads*maxBoxesPerThread];
		workLen = new int[numThreads];
	
		tempBoxes = new double[(2*dim+3) * numThreads*maxBoxesPerThread];
		tempWorkLen = new int[numThreads];
		
		outfile << maxBoxesPerThread << "\t";
		
		printf("..........................\n");
		printf("Stage 1: \n");
		printf("\tDimension: %d\n",dim);
		printf("\tNumber of threads: %d\n",numThreads);
		printf("\tMax number of boxes for thread: %d\n",maxBoxesPerThread);
		
		
		printf("Initializing boxes version 1\n");
		initializeBoxes_v1(boxes, workLen, numThreads, maxBoxesPerThread, dim);
		
		memcpy(tempBoxes,boxes,sizeof(double)*(2*dim+3) * numThreads*maxBoxesPerThread);
		memcpy(tempWorkLen,workLen,sizeof(int)*numThreads);	
		printf("Testing balancing on CPU (version 1)\n");
		balancingInfo = balancingOnCPU_v1(tempBoxes, tempWorkLen, numThreads, maxBoxesPerThread, dim);
		printf("numberOfMemoryCopies = %d\n",balancingInfo.numberOfMemoryCopies);
		printf("time = %f microseconds\n",balancingInfo.time);
		printf("numAllBoxes = %d\n",balancingInfo.numAllBoxes);
		printf("numAverageBoxes = %d\n",balancingInfo.numAverageBoxes);
		
		outfile << balancingInfo.time << "\t";
		
		printf("\n\n");
		
		
		memcpy(tempBoxes,boxes,sizeof(double)*(2*dim+3) * numThreads*maxBoxesPerThread);
		memcpy(tempWorkLen,workLen,sizeof(int)*numThreads);	
		printf("\nTesting balancing on CPU (version 2)\n");
		balancingInfo = balancingOnCPU_v2(tempBoxes, tempWorkLen, numThreads, maxBoxesPerThread, dim);
		printf("numberOfMemoryCopies = %d\n",balancingInfo.numberOfMemoryCopies);
		printf("time = %f microseconds\n",balancingInfo.time);
		printf("numAllBoxes = %d\n",balancingInfo.numAllBoxes);
		printf("numAverageBoxes = %d\n",balancingInfo.numAverageBoxes);
		
		outfile << balancingInfo.time << "\t";

		printf("\n\n");
		
		memcpy(tempBoxes,boxes,sizeof(double)*(2*dim+3) * numThreads*maxBoxesPerThread);
		memcpy(tempWorkLen,workLen,sizeof(int)*numThreads);	
		
		printf("\nTesting balancing on GPU (version 1)\n");
		balancingInfo = balancingOnGPU_v1(tempBoxes, tempWorkLen, numThreads, maxBoxesPerThread, dim);
		printf("numberOfMemoryCopies = %d\n",balancingInfo.numberOfMemoryCopies);
		printf("time = %f milliseconds\n",balancingInfo.time);
		printf("numAllBoxes = %d\n",balancingInfo.numAllBoxes);
		printf("numAverageBoxes = %d\n",balancingInfo.numAverageBoxes);
		
		outfile << balancingInfo.time << "\t";
		
		printf("\n\n");

		
		memcpy(tempBoxes,boxes,sizeof(double)*(2*dim+3) * numThreads*maxBoxesPerThread);
		memcpy(tempWorkLen,workLen,sizeof(int)*numThreads);	
		


		printf("\nTesting balancing on GPU (version 2)\n");
		balancingInfo = balancingOnGPU_v2(tempBoxes, tempWorkLen, numThreads, maxBoxesPerThread, dim);
		printf("numberOfMemoryCopies = %d\n",balancingInfo.numberOfMemoryCopies);
		printf("time = %f milliseconds\n",balancingInfo.time);
		printf("numAllBoxes = %d\n",balancingInfo.numAllBoxes);
		printf("numAverageBoxes = %d\n",balancingInfo.numAverageBoxes);
		
		outfile << balancingInfo.time << "\t";
		
		printf("\n\n");

		
		
		printf("..........................\n");
		
		
		
		printf("..........................\n");
		printf("Stage 2: \n");
		printf("\tDimension: %d\n",dim);
		printf("\tNumber of threads: %d\n",numThreads);
		printf("\tMax number of boxes for thread: %d\n",maxBoxesPerThread);
		
		
		printf("Initializing boxes version 2\n");
		initializeBoxes_v2(boxes, workLen, numThreads, maxBoxesPerThread, dim);
		
		
		memcpy(tempBoxes,boxes,sizeof(double)*(2*dim+3) * numThreads*maxBoxesPerThread);
		memcpy(tempWorkLen,workLen,sizeof(int)*numThreads);	

		
		printf("Testing balancing on CPU (version 3)\n");
		balancingInfo = balancingOnCPU_v3(tempBoxes, tempWorkLen, numThreads, maxBoxesPerThread, dim);
		printf("numberOfMemoryCopies = %d\n",balancingInfo.numberOfMemoryCopies);
		printf("time = %f microseconds\n",balancingInfo.time);
		printf("numAllBoxes = %d\n",balancingInfo.numAllBoxes);
		printf("numAverageBoxes = %d\n",balancingInfo.numAverageBoxes);
		
		outfile << balancingInfo.time << "\t";
		
		printf("\n\n");
		
		
		memcpy(tempBoxes,boxes,sizeof(double)*(2*dim+3) * numThreads*maxBoxesPerThread);
		memcpy(tempWorkLen,workLen,sizeof(int)*numThreads);	
		
		printf("\nTesting balancing on GPU (version 3)\n");
		balancingInfo = balancingOnGPU_v3(tempBoxes, tempWorkLen, numThreads, maxBoxesPerThread, dim);
		printf("numberOfMemoryCopies = %d\n",balancingInfo.numberOfMemoryCopies);
		printf("time = %f milliseconds\n",balancingInfo.time);
		printf("numAllBoxes = %d\n",balancingInfo.numAllBoxes);
		printf("numAverageBoxes = %d\n",balancingInfo.numAverageBoxes);
		
		outfile << balancingInfo.time << "\t";
		
		printf("\n\n");
	
		outfile << "\n"; 
		
		delete [] boxes;
		delete [] workLen;
		delete [] tempBoxes;
		delete [] tempWorkLen;
		
	}
	

	outfile.close();
	
	
	printf("..........................\n");

	return 0;
}
