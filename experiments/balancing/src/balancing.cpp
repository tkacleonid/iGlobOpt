
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





__global__ void globOptCUDA_2(double *boxes, const int dim, int *workLen)
{
	__shared__ double min_s[BLOCK_SIZE];
	__shared__ int workLen_s[BLOCK_SIZE];
	__shared__ int count[BLOCK_SIZE];
	
	double minRec = inRec;
	int i, j,bInd, hInd, n;
	double  h;
	
	int threadId = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	
	workLen_s[threadIdx.x] = workLen[threadId];
	min_s[threadIdx.x] = minRec;
	
	count[threadIdx.x] = 0;
	
	int half;
	
	
	
	__syncthreads();	
			

	n = 0;
	
	while(workLen_s[threadIdx.x] < BLOCK_SIZE && count[threadIdx.x] < MAX_GPU_ITER)
	{
		if(workLen_s[threadIdx.x] > 0)
		{
			
			bInd = threadId*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[threadIdx.x] - 1)*(2*inRank+3);
			fnCalcFunLimitsStyblinski_CUDA(inBox + bInd, inRank);
			
			if(min_s[threadIdx.x] > inBox[bInd + 2*inRank + 2])
			{
				min_s[threadIdx.x] = inBox[bInd + 2*inRank + 2];
			}
	
			if(min_s[threadIdx.x] - inBox[bInd + 2*inRank] < inEps)
			{
				--workLen_s[threadIdx.x];
				n++;
			}
			else
			{
				

				for(int j = 0; j < 16; j++)
				{
				bInd = threadId*1024*(2*inRank+3) + (workLen_s[threadIdx.x] - 1)*(2*inRank+3);
					
				hInd = 0;
				h = inBox[bInd + 1] - inBox[bInd];
				
				for(i = 0; i < inRank; i++)
				{
					if( h < inBox[bInd + i*2 + 1] - inBox[bInd + i*2]) 
					{
						h = inBox[bInd + i*2 + 1] - inBox[bInd + i*2];
						hInd = i;
					}
				}
				for(i = 0; i < inRank; i++)
				{
					if(i == hInd) 
					{
						inBox[bInd + i*2 + 1] = inBox[bInd + i*2] + h/2.0;
						inBox[bInd + 2*inRank + 3 + i*2] = inBox[bInd + i*2] + h/2.0;
						inBox[bInd + 2*inRank + 3 + i*2 + 1] = inBox[bInd + i*2] + h;
					}
					else
					{
						inBox[bInd + 2*inRank + 3 + i*2] = inBox[bInd + i*2];
						inBox[bInd + 2*inRank + 3 + i*2 + 1] = inBox[bInd + i*2 + 1];
					}
				}
				++workLen_s[threadIdx.x];
				}
			}
		}
			
		

		
		__syncthreads();
		
		if((threadIdx.x == 0))// && (count[threadIdx.x]+1) % 10 == 0)
		{
			for(i = 0; i < BLOCK_SIZE; i++)
			{
				if(minRec > min_s[i])
				{
					minRec = min_s[i];
				}
			}
		}
		
		__syncthreads();
		
		min_s[threadIdx.x] = minRec;	
		
		
		
		/*
		workLen_s_temp[threadId] = workLen[threadId];
		
		__syncthreads();
		
			
		
			
		if(workLen_s[threadId] == 0)
		{
			for(i = 0; i < 1024; i++)
			{
				if(workLen_s_temp[i] > 6 && workLen_s_temp[threadId] == 0)
				{
					atomicAdd(workLen_s_temp + i, -3);
					memcpy(inBox + bInd, inBox + i*1024*(2*inRank+3) + (workLen_s_temp[i] - 1)*(2*inRank+3), sizeof(double)*(2*inRank+3)*3);
					workLen_s_temp[threadId] += 3;
					break;
				}
			}
		}
			
			//workLen[threadId] = workLen_s_temp[threadId];
		__syncthreads();
			
		workLen_s[threadId] = workLen_s_temp[threadId];
			
			*/		
			
		__syncthreads();	
		
		/*
		
		
		if(threadIdx.x == 0 && (count[threadIdx.x]+1) % MAX_ITER_BEFORE_BALANCE == 0)
		{
			for(i = 0; i < BLOCK_SIZE; i++)
			{
				if(workLen_s[i] == 0)
				{
					for(j = 0; j < BLOCK_SIZE; j++)
					{
						if(workLen_s[j] > BORDER_BALANCE)
						{
							half = workLen_s[j]/2;
							workLen_s[j] -= half;
								memcpy(inBox + (i+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3), inBox + (j+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*half);
								workLen_s[i] += half;
								break;
						}
					}
				}		
			}	
		}	
		
		
		
		
		
		
		*/
		
		int numWorkBoxes = 0;
		int averageBoxesPerThread = 0;
		int curThreadWeTakeBoxesIndex = -1;
		int curThreadWeTakeBoxesCount = 0;
		int numBoxesWeTake = 0;
		int boxIndex = 0;
		if(threadIdx.x == 0 && (count[threadIdx.x]+1) % MAX_ITER_BEFORE_BALANCE == 0)
		{
			for(i = 0; i < BLOCK_SIZE; i++)
			{
				numWorkBoxes += workLen_s[i]; 	
			}
			averageBoxesPerThread = numWorkBoxes / BLOCK_SIZE;
			
			if(averageBoxesPerThread == 0) averageBoxesPerThread = averageBoxesPerThread + 1;
			
			curThreadWeTakeBoxesIndex = 0;
			for(i = 0; i < BLOCK_SIZE; i++)
			{
				if(workLen_s[i] < averageBoxesPerThread)
				{
					for(j = curThreadWeTakeBoxesIndex; j < BLOCK_SIZE; j++)
					{
						if(workLen_s[j] > averageBoxesPerThread)
						{
							
							numBoxesWeTake = averageBoxesPerThread - workLen_s[i] <= workLen_s[j] - averageBoxesPerThread ? averageBoxesPerThread - workLen_s[i] : workLen_s[j] - averageBoxesPerThread;
							workLen_s[j] -= numBoxesWeTake;
							memcpy(inBox + (i+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[i])*(2*inRank+3), inBox + (j+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*numBoxesWeTake);
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
			
			
			boxIndex = 0;
			for(i = 0; i < BLOCK_SIZE; i++)
			{
				if(workLen_s[i] == averageBoxesPerThread)
				{
					for(j = curThreadWeTakeBoxesIndex; j < BLOCK_SIZE; j++)
					{
						if(workLen_s[j] > averageBoxesPerThread + 1)
						{
							numBoxesWeTake = 1;
							workLen_s[j] -= numBoxesWeTake;
							memcpy(inBox + (i+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[i])*(2*inRank+3), inBox + (j+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*numBoxesWeTake);
							workLen_s[i] += numBoxesWeTake;	
							break;
						}
						
					}
					curThreadWeTakeBoxesIndex = j;
				}
				if(curThreadWeTakeBoxesIndex == BLOCK_SIZE - 1 && workLen_s[curThreadWeTakeBoxesIndex] <= averageBoxesPerThread + 1)
				{
					break;
				}
			}
			
			
			

			
		}
			
		
		
		__syncthreads();
		
		

		++count[threadIdx.x];

		
	}
	
	workLen[threadId] = workLen_s[threadIdx.x];
	min[threadId] = min_s[threadIdx.x];
	workCounts[threadId]=n;
	
}




