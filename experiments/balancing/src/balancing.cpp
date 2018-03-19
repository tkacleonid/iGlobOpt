
#include "balancing.hpp"



void sortQuickRecursive(int *indexes,int *ar,  const int n) {
   quickSortBase(indexes,ar,0,n-1);
}


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




__device__ void sortQuickRecursiveGPU(int *indexes,int *ar,  const int n) {
   quickSortBaseGPU(indexes,ar,0,n-1);
}

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



void initializeBoxes(double* boxes, int *workLen, int n, int m, int dim)
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





BalancingInfo balancingOnCPU_v2(double* boxes, int *workLen, int n, int m, int dim)
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

BalancingInfo balancingOnCPU_v3(double* boxes, int *workLen, int n, int m, int dim)
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
		memcpy(boxes + curThreadWeGiveBoxesIndex*m*(2*dim+3) + (workLen[curThreadWeGiveBoxesIndex])*(2*dim+3), boxes + curThreadWeTakeBoxesIndex*m*(2*dim+3) + (workLen[curThreadWeTakeBoxesIndex])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
		workLen[curThreadWeGiveBoxesIndex] += numBoxesWeTake;
		countMemoryCopies++;
			
	}
			
	auto end = std::chrono::high_resolution_clock::now();
	
	balancingInfo.time = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count();
	balancingInfo.numberOfMemoryCopies = countMemoryCopies;
	
	return balancingInfo;
				
}



BalancingInfo balancingOnGPU_v1(double* boxes, int *workLen, int n, int m, int dim)
{
				
	int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = 0;
	int numBoxesWeTake = 0;
	int i,j;
	//int countMemoryCopies = 0;
	int countAverageBoxesPerThreadMore = 0;
	
	BalancingInfo balancingInfo;
	balancingInfo.numThreads = n;
	balancingInfo.maxNumberOfBoxesPerThread = m;
	balancingInfo.version = WITHOUT_SORT_ON_GPU;
	
	
	
	double *dev_boxes = 0;
	int *dev_workLen = 0;
	int *dev_countMemoryCopies = 0;


	int GridSize = 1;
	int numThreads = n;
	int sizeInBox = numThreads*(dim*2+3)*sizeof(double)*m;
	
	float time, timeAll;
	
	int *countMemoryCopies = new int[numThreads*sizeof(int)];
	
	for(i = 0; i < numThreads; i++)
	{
		countMemoryCopies[i] = 0;
	}

	cudaEvent_t start, stop;
	
	for (i = 0; i < n; i++) {
		numWorkBoxes += workLen[i]; 	
	}
		
	averageBoxesPerThread = numWorkBoxes / n;	
	countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*n;
	
	balancingInfo.numAllBoxes = numWorkBoxes;
	balancingInfo.numAverageBoxes = countAverageBoxesPerThreadMore;

	auto start1 = std::chrono::high_resolution_clock::now();
	
	CHECKED_CALL(cudaSetDevice(0));
	CHECKED_CALL(cudaDeviceReset());
    CHECKED_CALL(cudaMalloc((void **)&dev_boxes, sizeInBox));
	CHECKED_CALL(cudaMalloc((void **)&dev_workLen, numThreads*sizeof(int)));	
	CHECKED_CALL(cudaMalloc((void **)&dev_countMemoryCopies, numThreads*sizeof(int)));
	
	
	CHECKED_CALL(cudaEventCreate(&start));
	CHECKED_CALL(cudaEventCreate(&stop));
	CHECKED_CALL(cudaMemcpy(dev_boxes, boxes, n*(2*dim+3)*sizeof(double)*m, cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_workLen, workLen, numThreads*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_countMemoryCopies, countMemoryCopies, numThreads*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaEventRecord(start, 0));

	balancingCUDA_v1<<<GridSize, n>>>(dev_boxes, dim, dev_workLen, dev_countMemoryCopies, m);
			
	CHECKED_CALL(cudaGetLastError());

	CHECKED_CALL(cudaEventRecord(stop, 0));
	CHECKED_CALL(cudaDeviceSynchronize());

	CHECKED_CALL(cudaMemcpy(boxes, dev_boxes, n*(2*dim+3)*sizeof(double)*m, cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(workLen, dev_workLen, numThreads*sizeof(int), cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(countMemoryCopies, dev_countMemoryCopies, numThreads*sizeof(int), cudaMemcpyDeviceToHost));

	CHECKED_CALL(cudaEventElapsedTime(&time, start, stop));

	CHECKED_CALL(cudaEventDestroy(start));
	CHECKED_CALL(cudaEventDestroy(stop));
		
	CHECKED_CALL(cudaFree(dev_boxes));
    CHECKED_CALL(cudaFree(dev_workLen));
	CHECKED_CALL(cudaFree(dev_countMemoryCopies));
	

	auto end = std::chrono::high_resolution_clock::now();
	
	balancingInfo.time = time;//(std::chrono::duration_cast<std::chrono::microseconds>(end - start1)).count();
	balancingInfo.numberOfMemoryCopies = countMemoryCopies[0];
	
	return balancingInfo;
				
}

BalancingInfo balancingOnGPU_v2(double* boxes, int *workLen, int n, int m, int dim)
{
				
	int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = 0;
	int numBoxesWeTake = 0;
	int i,j;
	//int countMemoryCopies = 0;
	int countAverageBoxesPerThreadMore = 0;
	
	BalancingInfo balancingInfo;
	balancingInfo.numThreads = n;
	balancingInfo.maxNumberOfBoxesPerThread = m;
	balancingInfo.version = WITHOUT_SORT_ON_GPU;
	
	
	
	double *dev_boxes = 0;
	int *dev_workLen = 0;
	int *dev_countMemoryCopies = 0;


	int GridSize = 1;
	int numThreads = n;
	int sizeInBox = numThreads*(dim*2+3)*sizeof(double)*m;
	
	float time, timeAll;
	
	int *countMemoryCopies = new int[numThreads*sizeof(int)];
	
	for(i = 0; i < numThreads; i++)
	{
		countMemoryCopies[i] = 0;
	}

	cudaEvent_t start, stop;
	
	for (i = 0; i < n; i++) {
		numWorkBoxes += workLen[i]; 	
	}
		
	averageBoxesPerThread = numWorkBoxes / n;	
	countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*n;
	
	balancingInfo.numAllBoxes = numWorkBoxes;
	balancingInfo.numAverageBoxes = countAverageBoxesPerThreadMore;

	auto start1 = std::chrono::high_resolution_clock::now();
	
	CHECKED_CALL(cudaSetDevice(0));
	CHECKED_CALL(cudaDeviceReset());
    CHECKED_CALL(cudaMalloc((void **)&dev_boxes, sizeInBox));
	CHECKED_CALL(cudaMalloc((void **)&dev_workLen, numThreads*sizeof(int)));	
	CHECKED_CALL(cudaMalloc((void **)&dev_countMemoryCopies, numThreads*sizeof(int)));
	
	
	CHECKED_CALL(cudaEventCreate(&start));
	CHECKED_CALL(cudaEventCreate(&stop));
	CHECKED_CALL(cudaMemcpy(dev_boxes, boxes, n*(2*dim+3)*sizeof(double)*m, cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_workLen, workLen, numThreads*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaMemcpy(dev_countMemoryCopies, countMemoryCopies, numThreads*sizeof(int), cudaMemcpyHostToDevice));
	CHECKED_CALL(cudaEventRecord(start, 0));

	balancingCUDA_v2<<<GridSize, n>>>(dev_boxes, dim, dev_workLen, dev_countMemoryCopies, m);
			
	CHECKED_CALL(cudaGetLastError());

	CHECKED_CALL(cudaEventRecord(stop, 0));
	CHECKED_CALL(cudaDeviceSynchronize());

	CHECKED_CALL(cudaMemcpy(boxes, dev_boxes, n*(2*dim+3)*sizeof(double)*m, cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(workLen, dev_workLen, numThreads*sizeof(int), cudaMemcpyDeviceToHost));
	CHECKED_CALL(cudaMemcpy(countMemoryCopies, dev_countMemoryCopies, numThreads*sizeof(int), cudaMemcpyDeviceToHost));

	CHECKED_CALL(cudaEventElapsedTime(&time, start, stop));

	CHECKED_CALL(cudaEventDestroy(start));
	CHECKED_CALL(cudaEventDestroy(stop));
		
	CHECKED_CALL(cudaFree(dev_boxes));
    CHECKED_CALL(cudaFree(dev_workLen));
	CHECKED_CALL(cudaFree(dev_countMemoryCopies));
	

	auto end = std::chrono::high_resolution_clock::now();
	
	balancingInfo.time = time;//(std::chrono::duration_cast<std::chrono::microseconds>(end - start1)).count();
	balancingInfo.numberOfMemoryCopies = countMemoryCopies[0];
	
	return balancingInfo;
				
}




__global__ void balancingCUDA_v1(double *boxes, const int dim, int *workLen, int *countMemoryCopies, const int m)
{
	__shared__ int workLen_s[1024];
	//__shared__ int countMemoryCopies[1024];
	
	int i, j;

	
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
	workLen_s[threadIdx.x] = workLen[threadId];	
	countMemoryCopies[threadIdx.x] = 0;
	
	__syncthreads();	


		
    int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = -1;
	int curThreadWeTakeBoxesCount = 0;
	int numBoxesWeTake = 0;
	int boxIndex = 0;
	if(threadIdx.x == 0)
	{
		for(i = 0; i < blockDim.x; i++)
		{
			numWorkBoxes += workLen_s[i]; 	
		}
		averageBoxesPerThread = numWorkBoxes / blockDim.x;
			
		if(averageBoxesPerThread == 0) averageBoxesPerThread = averageBoxesPerThread + 1;
			
		curThreadWeTakeBoxesIndex = 0;
		for(i = 0; i < blockDim.x; i++)
		{
			if(workLen_s[i] < averageBoxesPerThread)
			{
				for(j = curThreadWeTakeBoxesIndex; j < blockDim.x; j++)
				{
					if(workLen_s[j] > averageBoxesPerThread)
					{
							
						numBoxesWeTake = averageBoxesPerThread - workLen_s[i] <= workLen_s[j] - averageBoxesPerThread ? averageBoxesPerThread - workLen_s[i] : workLen_s[j] - averageBoxesPerThread;
						workLen_s[j] -= numBoxesWeTake;
						memcpy(boxes + (i+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[i])*(2*dim+3), boxes + (j+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[j])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
						countMemoryCopies[threadIdx.x] = countMemoryCopies[threadIdx.x] + 1;
						workLen_s[i] += numBoxesWeTake;	
						if(workLen_s[i] == averageBoxesPerThread) 
						{
							break;	
						}
					}
						
				}
				curThreadWeTakeBoxesIndex = j;
			}
				
		}
			
			
		for(i = 0; i < blockDim.x; i++)
		{
			if(workLen_s[i] == averageBoxesPerThread)
			{
				for(j = curThreadWeTakeBoxesIndex; j < blockDim.x; j++)
				{
					if(workLen_s[j] > averageBoxesPerThread + 1)
					{
						numBoxesWeTake = 1;
						workLen_s[j] -= numBoxesWeTake;
						memcpy(boxes + (i+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[i])*(2*dim+3), boxes + (j+blockIdx.x * dim)*m*(2*dim+3) + (workLen_s[j])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
						workLen_s[i] += numBoxesWeTake;	
						break;
					}
						
				}
				curThreadWeTakeBoxesIndex = j;
			}
			if(curThreadWeTakeBoxesIndex == blockDim.x - 1 && workLen_s[curThreadWeTakeBoxesIndex] <= averageBoxesPerThread + 1)
			{
				break;
			}
		}
			
			
			
	}
			
				
	__syncthreads();
		
	
	workLen[threadId] = workLen_s[threadIdx.x];
	countMemoryCopies[threadId] = countMemoryCopies[threadIdx.x];
	
}



__global__ void balancingCUDA_v2(double *boxes, const int dim, int *workLen, int *countMemoryCopies, const int m)
{
	__shared__ int workLen_s[1024];
	//__shared__ int countMemoryCopies[1024];
	
	int i, j;

	
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
	workLen_s[threadIdx.x] = workLen[threadId];	
	countMemoryCopies[threadIdx.x] = 0;
	
	__shared__ int workLenIndexes[1024];

	workLenIndexes[i] = threadId;
	
	__syncthreads();	


	
	int n = 1024;
	
	


	
	
		
    int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = 0;
	int numBoxesWeTake  = 0;
	int countAverageBoxesPerThreadMore = 0;
	int curThreadWeGiveBoxesIndex = 0;
	if(threadIdx.x == 0)
	{
			for (i = 0; i < blockDim.x; i++) {
			numWorkBoxes += workLen_s[i]; 	
		}
		averageBoxesPerThread = numWorkBoxes / n;
					
		countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*n;
		
		
		curThreadWeTakeBoxesIndex = n-1;
		curThreadWeGiveBoxesIndex = 0;
		
		
				
		for (i = 0; i < n; i++) {
			for (j = i+1; j < n; j++) {
				if(workLen_s[i] < workLen_s[j]) {
					int temp = workLen_s[i];
					workLen_s[i] = workLen_s[j];
					workLen_s[j] = temp;
					
					temp = workLenIndexes[i];
					workLenIndexes[i] = workLenIndexes[j];
					workLenIndexes[j] = temp;
				}

			}
		}		
				
		//sortQuickRecursiveGPU(workLenIndexes,workLen_s,n);
		
		//countMemoryCopies = 0;
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
			memcpy(boxes + (curThreadWeGiveBoxesIndex+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[curThreadWeGiveBoxesIndex])*(2*dim+3), boxes + (curThreadWeTakeBoxesIndex+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[curThreadWeTakeBoxesIndex])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
			workLen_s[curThreadWeGiveBoxesIndex] += numBoxesWeTake;
			countMemoryCopies[threadIdx.x]++;
				
		}
		
		
	
	
	
	
		
		
	/*	
		
		
		
		for(i = 0; i < blockDim.x; i++)
		{
			numWorkBoxes += workLen_s[i]; 	
		}
		averageBoxesPerThread = numWorkBoxes / blockDim.x;
			
		if(averageBoxesPerThread == 0) averageBoxesPerThread = averageBoxesPerThread + 1;
			
		curThreadWeTakeBoxesIndex = 0;
		for(i = 0; i < blockDim.x; i++)
		{
			if(workLen_s[i] < averageBoxesPerThread)
			{
				for(j = curThreadWeTakeBoxesIndex; j < blockDim.x; j++)
				{
					if(workLen_s[j] > averageBoxesPerThread)
					{
							
						numBoxesWeTake = averageBoxesPerThread - workLen_s[i] <= workLen_s[j] - averageBoxesPerThread ? averageBoxesPerThread - workLen_s[i] : workLen_s[j] - averageBoxesPerThread;
						workLen_s[j] -= numBoxesWeTake;
						memcpy(boxes + (i+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[i])*(2*dim+3), boxes + (j+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[j])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
						countMemoryCopies[threadIdx.x] = countMemoryCopies[threadIdx.x] + 1;
						workLen_s[i] += numBoxesWeTake;	
						if(workLen_s[i] == averageBoxesPerThread) 
						{
							break;	
						}
					}
						
				}
				curThreadWeTakeBoxesIndex = j;
			}
				
		}
			
			
		for(i = 0; i < blockDim.x; i++)
		{
			if(workLen_s[i] == averageBoxesPerThread)
			{
				for(j = curThreadWeTakeBoxesIndex; j < blockDim.x; j++)
				{
					if(workLen_s[j] > averageBoxesPerThread + 1)
					{
						numBoxesWeTake = 1;
						workLen_s[j] -= numBoxesWeTake;
						memcpy(boxes + (i+blockIdx.x * blockDim.x)*m*(2*dim+3) + (workLen_s[i])*(2*dim+3), boxes + (j+blockIdx.x * dim)*m*(2*dim+3) + (workLen_s[j])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
						workLen_s[i] += numBoxesWeTake;	
						break;
					}
						
				}
				curThreadWeTakeBoxesIndex = j;
			}
			if(curThreadWeTakeBoxesIndex == blockDim.x - 1 && workLen_s[curThreadWeTakeBoxesIndex] <= averageBoxesPerThread + 1)
			{
				break;
			}
		}
	*/		
			
			
	}
			
				
	__syncthreads();
		
	
	workLen[threadId] = workLen_s[threadIdx.x];
	countMemoryCopies[threadId] = countMemoryCopies[threadIdx.x];
	
}




