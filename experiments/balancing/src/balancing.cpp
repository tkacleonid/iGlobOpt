/*
 * File:   balancing.cpp
 * Author: Leonid Tkachenko
 *
 * Created on Feb 19, 2018, 12:43 PM
 */


#include "balancing.hpp"


/**
*	Test time of GPU kernel runs
*	@param numRuns the number of cuda testing calls
*	@param gridSize CUDA grid's size
*	@param blockSize CUDA block's size
*/
void testGPUKernelRun(const int numRuns, dim3 gridSize, dim3 blockSize)
{
	
	CHECKED_CALL(cudaSetDevice(DEVICE));
	CHECKED_CALL(cudaDeviceReset());
	
	auto start = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < numRuns; i++)
	{
		testCUDARun<<<gridSize, blockSize>>>(0);
	}
	auto end = std::chrono::high_resolution_clock::now();
	printf("AverageTime without synchronize: %d microseconds\n", (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count()/numRuns);

	start = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < numRuns; i++)
	{
		CHECKED_CALL(cudaThreadSynchronize());
		testCUDARun<<<gridSize, blockSize>>>(0);
		CHECKED_CALL(cudaThreadSynchronize());
	}
	end = std::chrono::high_resolution_clock::now();
	printf("AverageTime with synchronize: %d microseconds\n", (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count()/numRuns);
	
	
}

/**
*	Test CUDA kernel for GPU kernel runs
*	@param boxes the test boxes
*/
__global__ void testCUDARun(double *boxes)
{
	//code
}

/**
*	Test Transfer data to device
*	@param numRuns the number of cuda testing calls
*	@param gridSize CUDA grid's size
*	@param blockSize CUDA block's size
*	@param dataVolume the volume of data for transering to CUDA
*	@param fileName file to save data
*	@param isToFile if we should save data to file
*/
void testGPUTransferDataToDevice(const int numRuns, dim3 gridSize, dim3 blockSize, long long dataVolume, char* fileName, bool isToFile)
{
	
	CHECKED_CALL(cudaSetDevice(DEVICE));
	CHECKED_CALL(cudaDeviceReset());
	
	double *boxes = (double*) malloc(dataVolume);
	double *dev_boxes;
	
	CHECKED_CALL(cudaMalloc((void **)&dev_boxes, dataVolume));
	
	
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < numRuns; i++) {
		CHECKED_CALL(cudaMemcpy(dev_boxes, boxes, dataVolume, cudaMemcpyHostToDevice));
	}
	auto end = std::chrono::high_resolution_clock::now();
	
	long long speed = (long long) dataVolume/((std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count()/(((double) numRuns)*1000000));
	if (isToFile) {
		std::ofstream outfile;
		outfile.open(fileName, std::ios_base::app);
		if (outfile.fail())
			throw std::ios_base::failure(std::strerror(errno));
		outfile << dataVolume << "\t" << speed << "\n";
		outfile.close();
	}
	printf("Speed to transfer data to Device: %lld byte/s\n", speed);

	CHECKED_CALL(cudaFree(dev_boxes));
	free(boxes);	
}

/**
*	Test transfer data from device
*	@param numRuns the number of cuda testing calls
*	@param gridSize CUDA grid's size
*	@param blockSize CUDA block's size
*	@param dataVolume the volume of data for transering to CUDA
*	@param fileName file to save data
*	@param isToFile if we should save data to file
*/
void testGPUTransferDataFromDevice(const int numRuns, dim3 gridSize, dim3 blockSize, long long dataVolume, char* fileName, bool isToFile)
{
	
	CHECKED_CALL(cudaSetDevice(DEVICE));
	CHECKED_CALL(cudaDeviceReset());
	
	double *boxes = (double*) malloc(dataVolume);
	double *dev_boxes;
	
	CHECKED_CALL(cudaMalloc((void **)&dev_boxes, dataVolume));
	CHECKED_CALL(cudaMemcpy(dev_boxes, boxes, dataVolume, cudaMemcpyHostToDevice));
	
	
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < numRuns; i++) {
		CHECKED_CALL(cudaMemcpy(boxes, dev_boxes, dataVolume, cudaMemcpyDeviceToHost));
	}
	auto end = std::chrono::high_resolution_clock::now();
	
	long long speed = (long long) dataVolume/((std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count()/(((double) numRuns)*1000000));
	if (isToFile) {
		std::ofstream outfile;
		outfile.open(fileName, std::ios_base::app);
		if (outfile.fail())
			throw std::ios_base::failure(std::strerror(errno));
		outfile << dataVolume << "\t" << speed << "\n";
		outfile.close();
	}
	printf("Speed to transfer data from Device: %lld byte/s\n", speed);

	CHECKED_CALL(cudaFree(dev_boxes));
	free(boxes);	
}

/**
*	Test GPU inner memory access
*	@param numRuns the number of cuda testing calls
*	@param gridSize CUDA grid's size
*	@param blockSize CUDA block's size
*	@param fileName file to save data
*	@param isToFile if we should save data to file
*	@param partSize the number of values to copy by one thread
*/
void testGPUMemoryAccess(const int numRuns, dim3 gridSize, dim3 blockSize, char* fileName, bool isToFile, int partSize)
{	
	CHECKED_CALL(cudaSetDevice(DEVICE));
	CHECKED_CALL(cudaDeviceReset());
	
	int numThreads = gridSize.x*gridSize.y*gridSize.z*blockSize.x*blockSize.y*blockSize.z;
	double *ar1 = (double*) malloc(numThreads*sizeof(double)*partSize);
	double *ar2 = (double*) malloc(numThreads*sizeof(double)*partSize);
	double *dev_ar1;
	double *dev_ar2;
	
	for (int i = 0; i < numThreads; i++) {
		ar1[i] = (rand() % (rand()+1))/(double) numThreads;
	}
	

	CHECKED_CALL(cudaMalloc((void **)&dev_ar1, numThreads*sizeof(double)*partSize));
	CHECKED_CALL(cudaMalloc((void **)&dev_ar2, numThreads*sizeof(double)*partSize));
	CHECKED_CALL(cudaMemcpy(dev_ar1, ar1, numThreads*sizeof(double)*partSize, cudaMemcpyHostToDevice));
	
	cudaEvent_t startCuda, stopCuda;
	float time;
	
	CHECKED_CALL(cudaEventCreate(&startCuda));
	CHECKED_CALL(cudaEventCreate(&stopCuda));

	CHECKED_CALL(cudaEventRecord(startCuda, 0));	
	for (int i = 0; i < numRuns; i++) {
		testCUDAMemoryAccessRunMultiThread_v2<<<gridSize, blockSize>>>(dev_ar1,dev_ar2,partSize);
	}
	CHECKED_CALL(cudaEventRecord(stopCuda, 0));
	CHECKED_CALL(cudaEventSynchronize(stopCuda));
	CHECKED_CALL(cudaMemcpy(ar2,dev_ar2, numThreads*sizeof(double)*partSize, cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaEventElapsedTime(&time, startCuda, stopCuda));
	CHECKED_CALL(cudaEventDestroy(startCuda));
	CHECKED_CALL(cudaEventDestroy(stopCuda));
	
	if (isToFile) {
		std::ofstream outfile;
		outfile.open(fileName, std::ios_base::app);
		if (outfile.fail())
			throw std::ios_base::failure(std::strerror(errno));
		outfile << numThreads << "\t" << partSize << "\t" << time << "\n";
		outfile.close();
	}
	printf("Time assign array:\t%d\t%f milliseconds\n", numThreads,time);

	CHECKED_CALL(cudaFree(dev_ar1));
	CHECKED_CALL(cudaFree(dev_ar2));
	free(ar1);
	free(ar2);
}

/**
*	Test CUDA kernel for GPU multiple thread memory access with memcpy
*	@param ar1 the test array copy from
*	@param ar2 the test array copy To
*	@param partSize the number of values to copy by one thread
*/
__global__ void testCUDAMemoryAccessRunMultiThread_v1(double *ar1, double *ar2, int partSize)
{
	int gridSizeX = blockDim.x * gridDim.x;
	int gridSizeY = blockDim.y * gridDim.y;

	
	
	int threadId = threadIdx.z*gridSizeY*gridSizeX + threadIdx.y*gridSizeX + threadIdx.x;
	

	memcpy(ar2 + threadId*partSize, ar1 + threadId*partSize, sizeof(double)*partSize);
}

/**
*	Test CUDA kernel for GPU multiple thread memory access with element copies
*	@param ar1 the test array copy from
*	@param ar2 the test array copy To
*	@param partSize the number of values to copy by one thread
*/
__global__ void testCUDAMemoryAccessRunMultiThread_v2(double *ar1, double *ar2, int partSize)
{
	int gridSizeX = blockDim.x * gridDim.x;
	int gridSizeY = blockDim.y * gridDim.y;

	
	
	int threadId = threadIdx.z*gridSizeY*gridSizeX + threadIdx.y*gridSizeX + threadIdx.x;
	
	for (int i = 0; i < partSize; i++) {
		ar2[threadId*partSize + i] = ar1[threadId*partSize + i];
	}
}

/**
*	Test CUDA kernel for GPU single thread memory access with memcpy
*	@param ar1 the test array copy from
*	@param ar2 the test array copy To
*	@param partSize the number of values to copy by one thread
*/
__global__ void testCUDAMemoryAccessRunSingleThread_v1(double *ar1, double *ar2, int partSize)
{
	int gridSizeX = blockDim.x * gridDim.x;
	int gridSizeY = blockDim.y * gridDim.y;
	int gridSizeZ = blockDim.z * gridDim.z;
	int gridSizeAll = gridSizeX*gridSizeY*gridSizeZ;
	
	
	int threadId = threadIdx.z*gridSizeY*gridSizeX + threadIdx.y*gridSizeX + threadIdx.x;
	
	if (threadId == 0) {
		for (int i = 0; i < gridSizeAll; i++) {
			memcpy(ar2 + i*partSize, ar1 + i*partSize, sizeof(double)*partSize);
		}
	}
}

/**
*	Test CUDA kernel for GPU single thread memory access without memcpy
*	@param ar1 the test array copy from
*	@param ar2 the test array copy To
*	@param partSize the number of values to copy by one thread
*/
__global__ void testCUDAMemoryAccessRunSingleThread_v2(double *ar1, double *ar2, int partSize)
{
	int gridSizeX = blockDim.x * gridDim.x;
	int gridSizeY = blockDim.y * gridDim.y;
	int gridSizeZ = blockDim.z * gridDim.z;
	int gridSizeAll = gridSizeX*gridSizeY*gridSizeZ;
	
	
	int threadId = threadIdx.z*gridSizeY*gridSizeX + threadIdx.y*gridSizeX + threadIdx.x;
	
	if (threadId == 0) {
		for (int i = 0; i < gridSizeAll; i++) {
			for (int j = 0; j < partSize; j++) {
				ar2[i*partSize + j] = ar1[i*partSize + j];
			}
		}
	}
}

/**
*	Quick sort algorithm for balancing
*	@param indexes the array of boxes' indexes before sorting
*	@param ar the array of work boxes numbers
*	@param n the number of elements in array
*/
void sortQuickRecursive(int *indexes,int *ar,  const int n) {
   quickSortBase(indexes,ar,0,n-1);
}

/**
*	Quick sort algorithm for balancing
*	@param indexes the array of boxes' indexes before sorting
*	@param ar the array of work boxes numbers
*	@param l left index of array
*	@param r right index of array
*/
void quickSortBase (int *indexes,int *ar, const int l, const int r) {
    int i = l, j = r;
    int pp[3] = { ar[l], ar[r], ar[(l+r)>>1]};
    int p = pp[0];
    if (pp[1] >= pp[0] && pp[1]<=pp[0]) p=pp[1];
    else if (pp[2] >= pp[0] && pp[2]<=pp[1]) p=pp[2];
    
    while (i <= j) {
        while (p > ar[i])
           i++;
        while (ar[j] > p)
           j--;
        if (i <= j) {
			int temp;
			temp = ar[i];
			ar[i] = ar[j];
			ar[j] = temp;
			

			temp = indexes[i];
			indexes[i] = indexes[j];
			indexes[j] = temp;

            i++;
            j--;
        }
    }

    if (l < j)
       quickSortBase(indexes,ar,l, j);
    if (i < r)
       quickSortBase(indexes,ar,i, r);
}

/**
*	Quick sort algorithm for balancing on GPU
*	@param indexes the array of boxes' indexes before sorting
*	@param ar the array of work boxes numbers
*	@param n the number of elements in array
*/
__device__ void sortQuickRecursiveGPU(int *indexes,int *ar,  const int n) {
   quickSortBaseGPU(indexes,ar,0,n-1);
}

/**
*	Quick sort algorithm for balancing on GPU
*	@param indexes the array of boxes' indexes before sorting
*	@param ar the array of work boxes numbers
*	@param l left index of array
*	@param r right index of array
*/
__device__ void quickSortBaseGPU (int *indexes,int *ar, const int l, const int r) {
    int i = l, j = r;
    int pp[3] = { ar[l], ar[r], ar[(l+r)>>1]};
    int p = pp[0];
    if (pp[1] >= pp[0] && pp[1]<=pp[0]) p=pp[1];
    else if (pp[2] >= pp[0] && pp[2]<=pp[1]) p=pp[2];
    
    while (i <= j) {
        while (p > ar[i])
           i++;
        while (ar[j] > p)
           j--;
        if (i <= j) {
			int temp;
			temp = ar[i];
			ar[i] = ar[j];
			ar[j] = temp;
			

			temp = indexes[i];
			indexes[i] = indexes[j];
			indexes[j] = temp;

            i++;
            j--;
        }
    }

    if (l < j)
       quickSortBaseGPU(indexes,ar,l, j);
    if (i < r)
       quickSortBaseGPU(indexes,ar,i, r);
}


/**
*	Initialize boxes to test balancing procedure
*	@param boxes the array of boxes
*	@param workLen the array of numbers of boxes
*	@param n the number of threads
*	@param m the maximum number of boxes per thread
*	@param dim the function dimension
*/
void initializeBoxes_v1(double* boxes, int *workLen, int n, int m, int dim)
{
	//Initialize random seed
	srand(time(NULL));
	
	for(int i = 0; i < n; i++)
	{
		workLen[i] = rand()%(m+1);
		for(int j = 0; j < workLen[i]; j++)
		{
			for(int k = 0; k < dim; k++)
			{
				boxes[(2*dim+3)*i*m + (2*dim+3)*j + 2*k] = (rand() % (m+1))/(double) n;
				boxes[(2*dim+3)*i*m + (2*dim+3)*j + 2*k + 1] = (rand() % (m+1))/(double) n;
			}
		}		
	}
}


/**
*	Initialize boxes to test balancing procedure version 2
*	@param boxes the array of boxes
*	@param workLen the array of numbers of boxes
*	@param n the number of threads
*	@param m the maximum number of boxes per thread
*	@param dim the function dimension
*/
void initializeBoxes_v2(double* boxes, int *workLen, int n, int m, int dim)
{
	//Initialize random seed
	srand(time(NULL));
	
	for(int i = 0; i < n; i++)
	{
		workLen[i] = rand()%(m+1);
		for(int j = 0; j < workLen[i]; j++)
		{
			for(int k = 0; k < dim; k++)
			{
				boxes[(2*dim+3)*j*n + (2*dim+3)*i + 2*k] = (rand() % (m+1))/(double) n;
				boxes[(2*dim+3)*j*n + (2*dim+3)*i + 2*k + 1] = (rand() % (m+1))/(double) n;
			}
		}		
	}
}

/**
*	Balancing on CPU version 1
*	@param boxes the array of boxes
*	@param workLen the array of numbers of boxes
*	@param n the number of threads
*	@param m the maximum number of boxes per thread
*	@param dim the function dimension
*/
BalancingInfo balancingOnCPU_v1(double* boxes, int *workLen, int n, int m, int dim)
{
				
	int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = 0;
	int numBoxesWeTake = 0;
	int i,j;
	int countMemoryCopies = 0;
	int countAverageBoxesPerThreadMore = 0;
	
	BalancingInfo balancingInfo;
	balancingInfo.numThreads = n;
	balancingInfo.maxNumberOfBoxesPerThread = m;
	balancingInfo.version = WITHOUT_SORT_ON_CPU;
	
	auto start = std::chrono::high_resolution_clock::now();
	for (i = 0; i < n; i++) {
		numWorkBoxes += workLen[i]; 	
	}		
	averageBoxesPerThread = numWorkBoxes / n;	
	countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*n;
	if(averageBoxesPerThread == 0) averageBoxesPerThread = averageBoxesPerThread + 1;
	
	balancingInfo.numAllBoxes = numWorkBoxes;
	balancingInfo.numAverageBoxes = countAverageBoxesPerThreadMore;
			
	curThreadWeTakeBoxesIndex = 0;
	countMemoryCopies = 0;
	for (i = 0; i < n; i++) {
		if (workLen[i] < averageBoxesPerThread) {
			for (j = curThreadWeTakeBoxesIndex; j < n; j++) {
				if (workLen[j] > averageBoxesPerThread) {		
					numBoxesWeTake = averageBoxesPerThread - workLen[i] <= workLen[j] - averageBoxesPerThread ? averageBoxesPerThread - workLen[i] : workLen[j] - averageBoxesPerThread;
					workLen[j] -= numBoxesWeTake;
					memcpy(boxes + i*m*(2*dim+3) + workLen[i]*(2*dim+3), boxes + j*m*(2*dim+3) + workLen[j]*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
					workLen[i] += numBoxesWeTake;	
					countMemoryCopies++;
					if (workLen[i] == averageBoxesPerThread) break;	
				}			
			}
			curThreadWeTakeBoxesIndex = j;
		}
				
	}						
	
	for (i = 0; i < n; i++) {
		if (workLen[i] == averageBoxesPerThread) {
			for (j = curThreadWeTakeBoxesIndex; j < n; j++) {
				if (workLen[j] > averageBoxesPerThread + 1) {
					numBoxesWeTake = 1;
					workLen[j] -= numBoxesWeTake;
					memcpy(boxes + i*m*(2*dim+3) + workLen[i]*(2*dim+3), boxes + j*m*(2*dim+3) + workLen[j]*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
					workLen[i] += numBoxesWeTake;	
					countMemoryCopies++;
					break;
				}
						
			}
			curThreadWeTakeBoxesIndex = j;
		}
		if (curThreadWeTakeBoxesIndex == n - 1 && workLen[curThreadWeTakeBoxesIndex] <= averageBoxesPerThread + 1) break;

	}		
	auto end = std::chrono::high_resolution_clock::now();
	
	balancingInfo.time = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count();
	balancingInfo.numberOfMemoryCopies = countMemoryCopies;
	
	return balancingInfo;
				
}

/**
*	Balancing on CPU version 2 (with sort)
*	@param boxes the array of boxes
*	@param workLen the array of numbers of boxes
*	@param n the number of threads
*	@param m the maximum number of boxes per thread
*	@param dim the function dimension
*/
BalancingInfo balancingOnCPU_v2(double* boxes, int *workLen, int n, int m, int dim)
{	
	int *workLenIndexes = new int[n];
	for (int i = 0; i < n; i++) {
		workLenIndexes[i] = i;
	}
	

	int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = 0;
	int numBoxesWeTake  = 0;
	int countAverageBoxesPerThreadMore = 0;
	int curThreadWeGiveBoxesIndex = 0;
	int giveIndex = 0;
	int takeIndex = 0;
	int i = 0;
	int countMemoryCopies = 0;
	
	BalancingInfo balancingInfo;
	balancingInfo.numThreads = n;
	balancingInfo.maxNumberOfBoxesPerThread = m;
	balancingInfo.version = WITH_SORT_ON_CPU;
	
	
	auto start = std::chrono::high_resolution_clock::now();
	

	for (i = 0; i < n; i++) {
		numWorkBoxes += workLen[i]; 	
	}
	averageBoxesPerThread = numWorkBoxes / n;
				
	countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*n;
	
	balancingInfo.numAllBoxes = numWorkBoxes;
	balancingInfo.numAverageBoxes = countAverageBoxesPerThreadMore;
	
	curThreadWeTakeBoxesIndex = n-1;
	curThreadWeGiveBoxesIndex = 0;
			
	sortQuickRecursive(workLenIndexes,workLen,n);
	
	countMemoryCopies = 0;
	while (curThreadWeTakeBoxesIndex > curThreadWeGiveBoxesIndex) {
		if (workLen[curThreadWeTakeBoxesIndex] == averageBoxesPerThread) {
			curThreadWeTakeBoxesIndex--;
			continue;
		}
		if (workLen[curThreadWeTakeBoxesIndex] == averageBoxesPerThread + 1 && countAverageBoxesPerThreadMore > 0) {
			curThreadWeTakeBoxesIndex--;
			countAverageBoxesPerThreadMore--;
			continue;
		}
		
		if (workLen[curThreadWeGiveBoxesIndex] == averageBoxesPerThread && countAverageBoxesPerThreadMore == 0) {
			curThreadWeGiveBoxesIndex++;
			continue;
		}
		if (workLen[curThreadWeGiveBoxesIndex] == averageBoxesPerThread + 1 && countAverageBoxesPerThreadMore > 0) {
			curThreadWeGiveBoxesIndex++;
			countAverageBoxesPerThreadMore--;
			continue;
		}
		if (countAverageBoxesPerThreadMore > 1) {
			numBoxesWeTake = averageBoxesPerThread + 1 - workLen[curThreadWeGiveBoxesIndex] <= workLen[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread+1) 
							? averageBoxesPerThread + 1 - workLen[curThreadWeGiveBoxesIndex] 
							: workLen[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread + 1);
		}
		else if (countAverageBoxesPerThreadMore > 0) {
			numBoxesWeTake = averageBoxesPerThread + 1 - workLen[curThreadWeGiveBoxesIndex] <= workLen[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread) 
							? averageBoxesPerThread + 1 - workLen[curThreadWeGiveBoxesIndex] 
							: workLen[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread);
		}
		else {
			numBoxesWeTake = averageBoxesPerThread - workLen[curThreadWeGiveBoxesIndex] <= workLen[curThreadWeTakeBoxesIndex] - averageBoxesPerThread 
							? averageBoxesPerThread - workLen[curThreadWeGiveBoxesIndex] 
							: workLen[curThreadWeTakeBoxesIndex] - averageBoxesPerThread;
		}
		
		workLen[curThreadWeTakeBoxesIndex] -= numBoxesWeTake;
		
		//поменять curThreadWeGiveBoxesIndex на workLenIndexes
		giveIndex = workLenIndexes[curThreadWeGiveBoxesIndex];
		takeIndex = workLenIndexes[curThreadWeTakeBoxesIndex];
		memcpy(boxes + giveIndex*m*(2*dim+3) + (workLen[curThreadWeGiveBoxesIndex])*(2*dim+3), boxes + takeIndex*m*(2*dim+3) + (workLen[curThreadWeTakeBoxesIndex])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
		workLen[curThreadWeGiveBoxesIndex] += numBoxesWeTake;
		countMemoryCopies++;
			
	}
			
	auto end = std::chrono::high_resolution_clock::now();
	
	balancingInfo.time = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count();
	balancingInfo.numberOfMemoryCopies = countMemoryCopies;
	
	return balancingInfo;
				
}

/**
*	Balancing on CPU version 3
*	@param boxes the array of boxes
*	@param workLen the array of numbers of boxes
*	@param n the number of threads
*	@param m the maximum number of boxes per thread
*	@param dim the function dimension
*/
BalancingInfo balancingOnCPU_v3(double* boxes, int *workLen, int n, int m, int dim)
{
				
	int *workLenIndexes = new int[n];
	for (int i = 0; i < n; i++) {
		workLenIndexes[i] = i;
	}
	

	int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = 0;
	int curThreadWeGiveBoxesIndex = 0;
	int takeIndex = 0;
	int numBoxesWeTake  = 0;
	int countAverageBoxesPerThreadMore = 0;
	int giveIndex = 0;
	int i = 0;
	int countMemoryCopies = 0;
	
	BalancingInfo balancingInfo;
	balancingInfo.numThreads = n;
	balancingInfo.maxNumberOfBoxesPerThread = m;
	balancingInfo.version = WITH_SORT_ON_CPU;
	
	
	auto start = std::chrono::high_resolution_clock::now();
	

	for (i = 0; i < n; i++) {
		numWorkBoxes += workLen[i]; 	
	}
	averageBoxesPerThread = numWorkBoxes / n;
				
	countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*n;
	
	balancingInfo.numAllBoxes = numWorkBoxes;
	balancingInfo.numAverageBoxes = countAverageBoxesPerThreadMore;
	
	curThreadWeTakeBoxesIndex = n-1;
	curThreadWeGiveBoxesIndex = 0;
			
	sortQuickRecursive(workLenIndexes,workLen,n);
	
	countMemoryCopies = 0;
	while (curThreadWeTakeBoxesIndex > curThreadWeGiveBoxesIndex) {
		if (workLen[curThreadWeTakeBoxesIndex] == averageBoxesPerThread) {
			curThreadWeTakeBoxesIndex--;
			continue;
		}
		if (workLen[curThreadWeTakeBoxesIndex] == averageBoxesPerThread + 1 && countAverageBoxesPerThreadMore > 0) {
			curThreadWeTakeBoxesIndex--;
			countAverageBoxesPerThreadMore--;
			continue;
		}
		
		if (workLen[curThreadWeGiveBoxesIndex] == averageBoxesPerThread && countAverageBoxesPerThreadMore == 0) {
			curThreadWeGiveBoxesIndex++;
			continue;
		}
		if (workLen[curThreadWeGiveBoxesIndex] == averageBoxesPerThread + 1 && countAverageBoxesPerThreadMore > 0) {
			curThreadWeGiveBoxesIndex++;
			countAverageBoxesPerThreadMore--;
			continue;
		}
		if (countAverageBoxesPerThreadMore > 1) {
			numBoxesWeTake = averageBoxesPerThread + 1 - workLen[curThreadWeGiveBoxesIndex] <= workLen[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread+1) 
							? averageBoxesPerThread + 1 - workLen[curThreadWeGiveBoxesIndex] 
							: workLen[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread + 1);
		}
		else if (countAverageBoxesPerThreadMore > 0) {
			numBoxesWeTake = averageBoxesPerThread + 1 - workLen[curThreadWeGiveBoxesIndex] <= workLen[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread) 
							? averageBoxesPerThread + 1 - workLen[curThreadWeGiveBoxesIndex] 
							: workLen[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread);
		}
		else {
			numBoxesWeTake = averageBoxesPerThread - workLen[curThreadWeGiveBoxesIndex] <= workLen[curThreadWeTakeBoxesIndex] - averageBoxesPerThread 
							? averageBoxesPerThread - workLen[curThreadWeGiveBoxesIndex] 
							: workLen[curThreadWeTakeBoxesIndex] - averageBoxesPerThread;
		}
		
		workLen[curThreadWeTakeBoxesIndex] -= numBoxesWeTake;
		
		//поменять curThreadWeGiveBoxesIndex на workLenIndexes
		takeIndex = workLenIndexes[curThreadWeTakeBoxesIndex];
		giveIndex = workLenIndexes[curThreadWeGiveBoxesIndex];
		for (int i = 0; i < numBoxesWeTake; i++) {
			for (int j = 0; j < (2*dim+3); j++) {
				boxes[(workLen[curThreadWeGiveBoxesIndex] + i)*n*(2*dim+3) + giveIndex*(2*dim+3) + j] = boxes[(workLen[curThreadWeTakeBoxesIndex] + i)*n*(2*dim+3) + takeIndex*(2*dim+3) + j];
			}
		}
		workLen[curThreadWeGiveBoxesIndex] += numBoxesWeTake;
		countMemoryCopies++;
			
	}
			
	auto end = std::chrono::high_resolution_clock::now();
	
	balancingInfo.time = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count();
	balancingInfo.numberOfMemoryCopies = countMemoryCopies;
	
	return balancingInfo;
				
}


/**
*	Balancing on GPU version 1 (without sort)
*	@param boxes the array of boxes
*	@param workLen the array of numbers of boxes
*	@param n the number of threads
*	@param m the maximum number of boxes per thread
*	@param dim the function dimension
*/
BalancingInfo balancingOnGPU_v1(double* boxes, int *workLen, int n, int m, int dim)
{
				
	int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int i;
	int countAverageBoxesPerThreadMore = 0;
	
	BalancingInfo balancingInfo;
	balancingInfo.numThreads = n;
	balancingInfo.maxNumberOfBoxesPerThread = m;
	balancingInfo.version = WITHOUT_SORT_ON_GPU;
	
	
	
	double *dev_boxes = 0;
	int *dev_workLen = 0;
	int *dev_countMemoryCopies = 0;


	int GridSize = 1;
	int sizeInBox = n*(dim*2+3)*sizeof(double)*m;
	
	float time;
	
	int *countMemoryCopies = new int[n*sizeof(int)];	
	for (i = 0; i < n; i++) {
		countMemoryCopies[i] = 0;
	}

	cudaEvent_t startCuda, stopCuda;
	
	for (i = 0; i < n; i++) {
		numWorkBoxes += workLen[i]; 	
	}
		
	averageBoxesPerThread = numWorkBoxes / n;	
	countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*n;
	
	balancingInfo.numAllBoxes = numWorkBoxes;
	balancingInfo.numAverageBoxes = countAverageBoxesPerThreadMore;
	
	CHECKED_CALL(cudaSetDevice(0));
	CHECKED_CALL(cudaDeviceReset());
    CHECKED_CALL(cudaMalloc((void **)&dev_boxes, sizeInBox));
	CHECKED_CALL(cudaMalloc((void **)&dev_workLen, n*sizeof(int)));	
	CHECKED_CALL(cudaMalloc((void **)&dev_countMemoryCopies, n*sizeof(int)));
	
	CHECKED_CALL(cudaEventCreate(&startCuda));
	CHECKED_CALL(cudaEventCreate(&stopCuda));
	CHECKED_CALL(cudaMemcpy(dev_boxes, boxes, n*(2*dim+3)*sizeof(double)*m, cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_workLen, workLen, n*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_countMemoryCopies, countMemoryCopies, n*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaEventRecord(startCuda, 0));

	balancingCUDA_v1<<<GridSize, n>>>(dev_boxes, dim, dev_workLen, dev_countMemoryCopies, m);
			
	CHECKED_CALL(cudaGetLastError());
	CHECKED_CALL(cudaEventRecord(stopCuda, 0));
	CHECKED_CALL(cudaDeviceSynchronize());

	CHECKED_CALL(cudaMemcpy(boxes, dev_boxes, n*(2*dim+3)*sizeof(double)*m, cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(workLen, dev_workLen, n*sizeof(int), cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(countMemoryCopies, dev_countMemoryCopies, n*sizeof(int), cudaMemcpyDeviceToHost));

	CHECKED_CALL(cudaEventElapsedTime(&time, startCuda, stopCuda));

	CHECKED_CALL(cudaEventDestroy(startCuda));
	CHECKED_CALL(cudaEventDestroy(stopCuda));
		
	CHECKED_CALL(cudaFree(dev_boxes));
    CHECKED_CALL(cudaFree(dev_workLen));
	CHECKED_CALL(cudaFree(dev_countMemoryCopies));

	
	balancingInfo.time = time;
	balancingInfo.numberOfMemoryCopies = countMemoryCopies[0];
	
	return balancingInfo;
				
}

/**
*	Balancing on GPU version 2 (with sort)
*	@param boxes the array of boxes
*	@param workLen the array of numbers of boxes
*	@param n the number of threads
*	@param m the maximum number of boxes per thread
*	@param dim the function dimension
*/
BalancingInfo balancingOnGPU_v2(double* boxes, int *workLen, int n, int m, int dim)
{
				
	int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int i;
	int countAverageBoxesPerThreadMore = 0;
	
	BalancingInfo balancingInfo;
	balancingInfo.numThreads = n;
	balancingInfo.maxNumberOfBoxesPerThread = m;
	balancingInfo.version = WITHOUT_SORT_ON_GPU;
	
	double *dev_boxes = 0;
	int *dev_workLen = 0;
	int *dev_countMemoryCopies = 0;


	int GridSize = 1;
	int sizeInBox = n*(dim*2+3)*sizeof(double)*m;
	
	float time;
	
	int *countMemoryCopies = new int[n*sizeof(int)];
	
	for (i = 0; i < n; i++) {
		countMemoryCopies[i] = 0;
	}

	cudaEvent_t startCuda, stopCuda;
	
	for (i = 0; i < n; i++) {
		numWorkBoxes += workLen[i]; 	
	}
		
	averageBoxesPerThread = numWorkBoxes / n;	
	countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*n;
	
	balancingInfo.numAllBoxes = numWorkBoxes;
	balancingInfo.numAverageBoxes = countAverageBoxesPerThreadMore;
	
	CHECKED_CALL(cudaSetDevice(0));
	CHECKED_CALL(cudaDeviceReset());
    CHECKED_CALL(cudaMalloc((void **)&dev_boxes, sizeInBox));
	CHECKED_CALL(cudaMalloc((void **)&dev_workLen, n*sizeof(int)));	
	CHECKED_CALL(cudaMalloc((void **)&dev_countMemoryCopies, n*sizeof(int)));
	
	
	CHECKED_CALL(cudaEventCreate(&startCuda));
	CHECKED_CALL(cudaEventCreate(&stopCuda));
	CHECKED_CALL(cudaMemcpy(dev_boxes, boxes, n*(2*dim+3)*sizeof(double)*m, cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_workLen, workLen, n*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_countMemoryCopies, countMemoryCopies, n*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaEventRecord(startCuda, 0));

	balancingCUDA_v2<<<GridSize, n>>>(dev_boxes, dim, dev_workLen, dev_countMemoryCopies, m);
			
	CHECKED_CALL(cudaGetLastError());
	CHECKED_CALL(cudaEventRecord(stopCuda, 0));
	CHECKED_CALL(cudaDeviceSynchronize());

	CHECKED_CALL(cudaMemcpy(boxes, dev_boxes, n*(2*dim+3)*sizeof(double)*m, cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(workLen, dev_workLen, n*sizeof(int), cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(countMemoryCopies, dev_countMemoryCopies, n*sizeof(int), cudaMemcpyDeviceToHost));

	CHECKED_CALL(cudaEventElapsedTime(&time, startCuda, stopCuda));
	
	CHECKED_CALL(cudaEventDestroy(startCuda));
	CHECKED_CALL(cudaEventDestroy(stopCuda));
		
	CHECKED_CALL(cudaFree(dev_boxes));
    CHECKED_CALL(cudaFree(dev_workLen));
	CHECKED_CALL(cudaFree(dev_countMemoryCopies));
	
	balancingInfo.time = time;
	balancingInfo.numberOfMemoryCopies = countMemoryCopies[0];
	
	return balancingInfo;
				
}

/**
*	Balancing on GPU version 3 (with sort)
*	@param boxes the array of boxes
*	@param workLen the array of numbers of boxes
*	@param n the number of threads
*	@param m the maximum number of boxes per thread
*	@param dim the function dimension
*/
BalancingInfo balancingOnGPU_v3(double* boxes, int *workLen, int n, int m, int dim)
{
				
	int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int i;
	int countAverageBoxesPerThreadMore = 0;
	
	BalancingInfo balancingInfo;
	balancingInfo.numThreads = n;
	balancingInfo.maxNumberOfBoxesPerThread = m;
	balancingInfo.version = WITHOUT_SORT_ON_GPU;
	
	double *dev_boxes = 0;
	int *dev_workLen = 0;
	int *dev_countMemoryCopies = 0;


	int GridSize = 1;
	int sizeInBox = n*(dim*2+3)*sizeof(double)*m;
	
	float time;
	
	int *countMemoryCopies = new int[n*sizeof(int)];
	
	for (i = 0; i < n; i++) {
		countMemoryCopies[i] = 0;
	}

	cudaEvent_t startCuda, stopCuda;
	
	for (i = 0; i < n; i++) {
		numWorkBoxes += workLen[i]; 	
	}
		
	averageBoxesPerThread = numWorkBoxes / n;	
	countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*n;
	
	balancingInfo.numAllBoxes = numWorkBoxes;
	balancingInfo.numAverageBoxes = countAverageBoxesPerThreadMore;
	
	CHECKED_CALL(cudaSetDevice(0));
	CHECKED_CALL(cudaDeviceReset());
    CHECKED_CALL(cudaMalloc((void **)&dev_boxes, sizeInBox));
	CHECKED_CALL(cudaMalloc((void **)&dev_workLen, n*sizeof(int)));	
	CHECKED_CALL(cudaMalloc((void **)&dev_countMemoryCopies, n*sizeof(int)));
	
	
	CHECKED_CALL(cudaEventCreate(&startCuda));
	CHECKED_CALL(cudaEventCreate(&stopCuda));
	CHECKED_CALL(cudaMemcpy(dev_boxes, boxes, n*(2*dim+3)*sizeof(double)*m, cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_workLen, workLen, n*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_countMemoryCopies, countMemoryCopies, n*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaEventRecord(startCuda, 0));

	balancingCUDA_v3<<<GridSize, n>>>(dev_boxes, dim, dev_workLen, dev_countMemoryCopies, m);
			
	CHECKED_CALL(cudaGetLastError());
	CHECKED_CALL(cudaEventRecord(stopCuda, 0));
	CHECKED_CALL(cudaDeviceSynchronize());

	CHECKED_CALL(cudaMemcpy(boxes, dev_boxes, n*(2*dim+3)*sizeof(double)*m, cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(workLen, dev_workLen, n*sizeof(int), cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(countMemoryCopies, dev_countMemoryCopies, n*sizeof(int), cudaMemcpyDeviceToHost));

	CHECKED_CALL(cudaEventElapsedTime(&time, startCuda, stopCuda));
	
	CHECKED_CALL(cudaEventDestroy(startCuda));
	CHECKED_CALL(cudaEventDestroy(stopCuda));
		
	CHECKED_CALL(cudaFree(dev_boxes));
    CHECKED_CALL(cudaFree(dev_workLen));
	CHECKED_CALL(cudaFree(dev_countMemoryCopies));
	
	balancingInfo.time = time;
	balancingInfo.numberOfMemoryCopies = countMemoryCopies[0];
	
	return balancingInfo;
				
}

/**
*	Balancing on GPU version 1 CUDA kernel(without sort)
*	@param boxes the array of boxes
*	@param workLen the array of numbers of boxes
*	@param n the number of threads
*	@param m the maximum number of boxes per thread
*	@param dim the function dimension
*/
__global__ void balancingCUDA_v1(double *boxes, const int dim, int *workLen, int *countMemoryCopies, const int m)
{
	__shared__ int workLen_s[BLOCK_SIZE];
	
	int i, j;
	
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
	workLen_s[threadIdx.x] = workLen[threadId];	
	countMemoryCopies[threadIdx.x] = 0;
	
	__syncthreads();	
		
    int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = -1;
	int numBoxesWeTake = 0;
	if (threadIdx.x == 0) {
		for (i = 0; i < blockDim.x; i++) {
			numWorkBoxes += workLen_s[i]; 	
		}
		averageBoxesPerThread = numWorkBoxes / blockDim.x;
			
		if(averageBoxesPerThread == 0) averageBoxesPerThread = averageBoxesPerThread + 1;
			
		curThreadWeTakeBoxesIndex = 0;
		for (i = 0; i < blockDim.x; i++) {
			if (workLen_s[i] < averageBoxesPerThread) {
				for (j = curThreadWeTakeBoxesIndex; j < blockDim.x; j++) {
					if (workLen_s[j] > averageBoxesPerThread) {
							
						numBoxesWeTake = averageBoxesPerThread - workLen_s[i] <= workLen_s[j] - averageBoxesPerThread ? averageBoxesPerThread - workLen_s[i] : workLen_s[j] - averageBoxesPerThread;
						workLen_s[j] -= numBoxesWeTake;
						
						int indTo = (i+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[i])*(2*dim+3);
						int indFrom = (j+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[j])*(2*dim+3);
						for (int d = 0; d < numBoxesWeTake*(2*dim+3); d++) {			
							boxes[indTo+d] = boxes[indFrom+d];
						}
						
						//memcpy(boxes + (i+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[i])*(2*dim+3), boxes + (j+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[j])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
						countMemoryCopies[threadIdx.x] = countMemoryCopies[threadIdx.x] + 1;
						workLen_s[i] += numBoxesWeTake;	
						if (workLen_s[i] == averageBoxesPerThread) {
							break;	
						}
					}
						
				}
				curThreadWeTakeBoxesIndex = j;
			}
				
		}
			
			
		for (i = 0; i < blockDim.x; i++) {
			if (workLen_s[i] == averageBoxesPerThread) {
				for (j = curThreadWeTakeBoxesIndex; j < blockDim.x; j++) {
					if (workLen_s[j] > averageBoxesPerThread + 1) {
						numBoxesWeTake = 1;
						workLen_s[j] -= numBoxesWeTake;
						
						int indTo = (i+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[i])*(2*dim+3);
						int indFrom = (j+blockIdx.x * dim)*m*(2*dim+3) + (workLen_s[j])*(2*dim+3);
						for (int d = 0; d < numBoxesWeTake*(2*dim+3); d++) {			
							boxes[indTo+d] = boxes[indFrom+d];
						}
						
						//memcpy(boxes + (i+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[i])*(2*dim+3), boxes + (j+blockIdx.x * dim)*m*(2*dim+3) + (workLen_s[j])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
						countMemoryCopies[threadIdx.x] = countMemoryCopies[threadIdx.x] + 1;
						workLen_s[i] += numBoxesWeTake;	
						break;
					}
						
				}
				curThreadWeTakeBoxesIndex = j;
			}
			if (curThreadWeTakeBoxesIndex == blockDim.x - 1 && workLen_s[curThreadWeTakeBoxesIndex] <= averageBoxesPerThread + 1) {
				break;
			}
		}		
	}
			
				
	__syncthreads();
		
	
	workLen[threadId] = workLen_s[threadIdx.x];
	countMemoryCopies[threadId] = countMemoryCopies[threadIdx.x];
	
}


/**
*	Balancing on GPU version 2 CUDA kernel(with sort)
*	@param boxes the array of boxes
*	@param workLen the array of numbers of boxes
*	@param n the number of threads
*	@param m the maximum number of boxes per thread
*	@param dim the function dimension
*/
__global__ void balancingCUDA_v2(double *boxes, const int dim, int *workLen, int *countMemoryCopies, const int m)
{
	__shared__ int workLen_s[BLOCK_SIZE];
	
	int i, j;	
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
	workLen_s[threadIdx.x] = workLen[threadId];	
	countMemoryCopies[threadIdx.x] = 0;
	
	__shared__ int workLenIndexes[BLOCK_SIZE];

	workLenIndexes[threadIdx.x] = threadIdx.x;
	
	__syncthreads();	
	
    int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = 0;
	int numBoxesWeTake  = 0;
	int countAverageBoxesPerThreadMore = 0;
	int curThreadWeGiveBoxesIndex = 0;
	int giveIndex = 0;
	int takeIndex = 0;
	if(threadIdx.x == 0)
	{
			for (i = 0; i < blockDim.x; i++) {
			numWorkBoxes += workLen_s[i]; 	
		}
		averageBoxesPerThread = numWorkBoxes / blockDim.x;
					
		countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*blockDim.x;
		
		
		curThreadWeTakeBoxesIndex = blockDim.x-1;
		curThreadWeGiveBoxesIndex = 0;

/*		
		for (i = 0; i < blockDim.x; i++) {
			for (j = i+1; j < blockDim.x; j++) {
				if(workLen_s[i] > workLen_s[j]) {
					int temp = workLen_s[i];
					workLen_s[i] = workLen_s[j];
					workLen_s[j] = temp;
					
					temp = workLenIndexes[i];
					workLenIndexes[i] = workLenIndexes[j];
					workLenIndexes[j] = temp;
				}
			}
		}		
*/	
		sortQuickRecursiveGPU(workLenIndexes,workLen_s,blockDim.x);
		
		while (curThreadWeTakeBoxesIndex > curThreadWeGiveBoxesIndex) {
			if (workLen_s[curThreadWeTakeBoxesIndex] == averageBoxesPerThread) {
				curThreadWeTakeBoxesIndex--;
				continue;
			}
			if (workLen_s[curThreadWeTakeBoxesIndex] == averageBoxesPerThread + 1 && countAverageBoxesPerThreadMore > 0) {
				curThreadWeTakeBoxesIndex--;
				countAverageBoxesPerThreadMore--;
				continue;
			}
			
			if (workLen_s[curThreadWeGiveBoxesIndex] == averageBoxesPerThread && countAverageBoxesPerThreadMore == 0) {
				curThreadWeGiveBoxesIndex++;
				continue;
			}
			if (workLen_s[curThreadWeGiveBoxesIndex] == averageBoxesPerThread + 1 && countAverageBoxesPerThreadMore > 0) {
				curThreadWeGiveBoxesIndex++;
				countAverageBoxesPerThreadMore--;
				continue;
			}
			if (countAverageBoxesPerThreadMore > 1) {
				numBoxesWeTake = averageBoxesPerThread + 1 - workLen_s[curThreadWeGiveBoxesIndex] <= workLen_s[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread+1) 
								? averageBoxesPerThread + 1 - workLen_s[curThreadWeGiveBoxesIndex] 
								: workLen_s[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread + 1);
			}
			else if (countAverageBoxesPerThreadMore > 0) {
				numBoxesWeTake = averageBoxesPerThread + 1 - workLen_s[curThreadWeGiveBoxesIndex] <= workLen_s[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread) 
								? averageBoxesPerThread + 1 - workLen_s[curThreadWeGiveBoxesIndex] 
								: workLen_s[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread);
			}
			else {
				numBoxesWeTake = averageBoxesPerThread - workLen_s[curThreadWeGiveBoxesIndex] <= workLen_s[curThreadWeTakeBoxesIndex] - averageBoxesPerThread 
								? averageBoxesPerThread - workLen_s[curThreadWeGiveBoxesIndex] 
								: workLen_s[curThreadWeTakeBoxesIndex] - averageBoxesPerThread;
			}
			
			workLen_s[curThreadWeTakeBoxesIndex] -= numBoxesWeTake;
			
			giveIndex = workLenIndexes[curThreadWeGiveBoxesIndex];
			takeIndex = workLenIndexes[curThreadWeTakeBoxesIndex];
			int indTo = (giveIndex+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[curThreadWeGiveBoxesIndex])*(2*dim+3);
			int indFrom = (takeIndex+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[curThreadWeTakeBoxesIndex])*(2*dim+3);
			for (int d = 0; d < numBoxesWeTake*(2*dim+3); d++) {
				boxes[indTo + d] = boxes[indFrom + d];
			}
			
			//memcpy(boxes + (curThreadWeGiveBoxesIndex+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[curThreadWeGiveBoxesIndex])*(2*dim+3), boxes + (curThreadWeTakeBoxesIndex+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[curThreadWeTakeBoxesIndex])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
			workLen_s[curThreadWeGiveBoxesIndex] += numBoxesWeTake;
			countMemoryCopies[threadIdx.x]++;
				
		}		
	}				
	__syncthreads();
		
	
	workLen[threadId] = workLen_s[threadIdx.x];
	countMemoryCopies[threadId] = countMemoryCopies[threadIdx.x];
	
}

/**
*	Balancing on GPU version 3 CUDA kernel(with sort)
*	@param boxes the array of boxes
*	@param workLen the array of numbers of boxes
*	@param n the number of threads
*	@param m the maximum number of boxes per thread
*	@param dim the function dimension
*/
__global__ void balancingCUDA_v3(double *boxes, const int dim, int *workLen, int *countMemoryCopies, const int m)
{
	__shared__ int workLen_s[BLOCK_SIZE];
	
	int i, j;	
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
	workLen_s[threadIdx.x] = workLen[threadId];	
	countMemoryCopies[threadIdx.x] = 0;
	
	__shared__ int workLenIndexes[BLOCK_SIZE];

	workLenIndexes[threadIdx.x] = threadIdx.x;
	
	__syncthreads();	
	
    int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = 0;
	int numBoxesWeTake  = 0;
	int countAverageBoxesPerThreadMore = 0;
	int curThreadWeGiveBoxesIndex = 0;
	int giveIndex = 0;
	int takeIndex = 0;
	if(threadIdx.x == 0)
	{
		for (i = 0; i < blockDim.x; i++) {
			numWorkBoxes += workLen_s[i]; 	
		}
		averageBoxesPerThread = numWorkBoxes / blockDim.x;
					
		countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*blockDim.x;
		
		
		curThreadWeTakeBoxesIndex = blockDim.x-1;
		curThreadWeGiveBoxesIndex = 0;
/*						
		for (i = 0; i < blockDim.x; i++) {
			for (j = i+1; j < blockDim.x; j++) {
				if(workLen_s[i] > workLen_s[j]) {
					int temp = workLen_s[i];
					workLen_s[i] = workLen_s[j];
					workLen_s[j] = temp;
					
					temp = workLenIndexes[i];
					workLenIndexes[i] = workLenIndexes[j];
					workLenIndexes[j] = temp;
				}
			}
		}		
*/	
		sortQuickRecursiveGPU(workLenIndexes,workLen_s,n);
		
		while (curThreadWeTakeBoxesIndex > curThreadWeGiveBoxesIndex) {
			if (workLen_s[curThreadWeTakeBoxesIndex] == averageBoxesPerThread) {
				curThreadWeTakeBoxesIndex--;
				continue;
			}
			if (workLen_s[curThreadWeTakeBoxesIndex] == averageBoxesPerThread + 1 && countAverageBoxesPerThreadMore > 0) {
				curThreadWeTakeBoxesIndex--;
				countAverageBoxesPerThreadMore--;
				continue;
			}
			
			if (workLen_s[curThreadWeGiveBoxesIndex] == averageBoxesPerThread && countAverageBoxesPerThreadMore == 0) {
				curThreadWeGiveBoxesIndex++;
				continue;
			}
			if (workLen_s[curThreadWeGiveBoxesIndex] == averageBoxesPerThread + 1 && countAverageBoxesPerThreadMore > 0) {
				curThreadWeGiveBoxesIndex++;
				countAverageBoxesPerThreadMore--;
				continue;
			}
			if (countAverageBoxesPerThreadMore > 1) {
				numBoxesWeTake = averageBoxesPerThread + 1 - workLen_s[curThreadWeGiveBoxesIndex] <= workLen_s[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread+1) 
								? averageBoxesPerThread + 1 - workLen_s[curThreadWeGiveBoxesIndex] 
								: workLen_s[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread + 1);
			}
			else if (countAverageBoxesPerThreadMore > 0) {
				numBoxesWeTake = averageBoxesPerThread + 1 - workLen_s[curThreadWeGiveBoxesIndex] <= workLen_s[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread) 
								? averageBoxesPerThread + 1 - workLen_s[curThreadWeGiveBoxesIndex] 
								: workLen_s[curThreadWeTakeBoxesIndex] - (averageBoxesPerThread);
			}
			else {
				numBoxesWeTake = averageBoxesPerThread - workLen_s[curThreadWeGiveBoxesIndex] <= workLen_s[curThreadWeTakeBoxesIndex] - averageBoxesPerThread 
								? averageBoxesPerThread - workLen_s[curThreadWeGiveBoxesIndex] 
								: workLen_s[curThreadWeTakeBoxesIndex] - averageBoxesPerThread;
			}
			
			workLen_s[curThreadWeTakeBoxesIndex] -= numBoxesWeTake;
			
			giveIndex = workLenIndexes[curThreadWeGiveBoxesIndex];
			takeIndex = workLenIndexes[curThreadWeTakeBoxesIndex];
			int indTo = (giveIndex+blockIdx.x * blockDim.x)*blockDim.x*(2*dim+3) + (workLen_s[curThreadWeGiveBoxesIndex])*(2*dim+3);
			int indFrom = (takeIndex+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[curThreadWeTakeBoxesIndex])*(2*dim+3);
			for (j = 0; j < numBoxesWeTake; j++) {
				for (i = 0; i < 2*dim +3; i++) {
					//boxes[indTo + j*blockDim.x*(2*dim+3) + i] = boxes[indFrom + j*blockDim.x*(2*dim+3) + i];
				}
			}
			
			//memcpy(boxes + (curThreadWeGiveBoxesIndex+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[curThreadWeGiveBoxesIndex])*(2*dim+3), boxes + (curThreadWeTakeBoxesIndex+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[curThreadWeTakeBoxesIndex])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
			workLen_s[curThreadWeGiveBoxesIndex] += numBoxesWeTake;
			countMemoryCopies[threadIdx.x]++;
				
		}		
	}				
	__syncthreads();
		
	
	workLen[threadId] = workLen_s[threadIdx.x];
	countMemoryCopies[threadId] = countMemoryCopies[threadIdx.x];
	
}





