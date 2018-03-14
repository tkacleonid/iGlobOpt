
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


BalancingInfo balancingOnCPU(double* boxes, int *workLen, int n, int m, int dim)
{
	
	printf("\n\n");
	for(int i = 0; i < n; i++)
	{		
		printf("%d\t", workLen[i]);	
	}
	printf("\n\n");
				
				
	
	int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = -1;
	int curThreadWeTakeBoxesCount = 0;
	int numBoxesWeTake = 0;
	int boxIndex = 0;
	int countAverageBoxesPerThreadMore = 0;
	int plusOne = 0;
	int stopMore;
	
	

	for(int i = 0; i < n; i++)
	{
		numWorkBoxes += workLen[i]; 	
	}
	
	averageBoxesPerThread = numWorkBoxes / n;		
	countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*n;
	
	curThreadWeTakeBoxesIndex = 0;
	
	printf("NumWorkBoxes: %d\n", numWorkBoxes);
	printf("countAverageBoxesPerThreadMore: %d\n", countAverageBoxesPerThreadMore);
	
	for(int i = 0; i < n; i++)
	{

		if(workLen[i] < averageBoxesPerThread+1 && countAverageBoxesPerThreadMore > 0)
		{
			
			for(int j = curThreadWeTakeBoxesIndex; j < n; j++)
			{
				//if(workLen[j] == averageBoxesPerThread + 1 && i > j 
				if(workLen[j] > averageBoxesPerThread+1)  
				{
					if	(countAverageBoxesPerThreadMore > 0) plusOne = 1;
					else plusOne = 0;
					numBoxesWeTake = (averageBoxesPerThread+plusOne) - workLen[i] <= workLen[j] - (averageBoxesPerThread+plusOne) ? (averageBoxesPerThread+plusOne) - workLen[i] : workLen[j] - (averageBoxesPerThread+plusOne);
					if(numBoxesWeTake + workLen[i] == averageBoxesPerThread + 1)  {
						countAverageBoxesPerThreadMore--;
						//if (countAverageBoxesPerThreadMore == 0)
					}
					
					workLen[j] -= numBoxesWeTake;
					//memcpy(boxes + i*m*(2*dim+3) + (workLen[i])*(2*dim+3), boxes + j*m*(2*dim+3) + (workLen[j])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
					workLen[i] += numBoxesWeTake;	
					
					if(workLen[j] == averageBoxesPerThread + 1 && countAverageBoxesPerThreadMore > 0)  {
						countAverageBoxesPerThreadMore--;
						//if (countAverageBoxesPerThreadMore == 0)
					}

					if((countAverageBoxesPerThreadMore == 0) || workLen[i] == averageBoxesPerThread + 1) 
					{
						curThreadWeTakeBoxesIndex = j;
						break;	
					}
				}
						
			}
			
		}
		if(countAverageBoxesPerThreadMore == 0) {
			stopMore = i;
			break;
		}
		
	}
	
	for(int i = stopMore; i < n; i++)
	{

		if(workLen[i] < averageBoxesPerThread+1 && countAverageBoxesPerThreadMore > 0)
		{
			
			for(int j = curThreadWeTakeBoxesIndex; j < n; j++)
			{
				//if(workLen[j] == averageBoxesPerThread + 1 && i > j 
				if(workLen[j] > averageBoxesPerThread+1)  
				{
					if	(countAverageBoxesPerThreadMore > 0) plusOne = 1;
					else plusOne = 0;
					numBoxesWeTake = (averageBoxesPerThread+plusOne) - workLen[i] <= workLen[j] - (averageBoxesPerThread+plusOne) ? (averageBoxesPerThread+plusOne) - workLen[i] : workLen[j] - (averageBoxesPerThread+plusOne);
					if(numBoxesWeTake + workLen[i] == averageBoxesPerThread + 1)  {
						countAverageBoxesPerThreadMore--;
						//if (countAverageBoxesPerThreadMore == 0)
					}
					
					workLen[j] -= numBoxesWeTake;
					//memcpy(boxes + i*m*(2*dim+3) + (workLen[i])*(2*dim+3), boxes + j*m*(2*dim+3) + (workLen[j])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
					workLen[i] += numBoxesWeTake;	
					
					if(workLen[j] == averageBoxesPerThread + 1 && countAverageBoxesPerThreadMore > 0)  {
						countAverageBoxesPerThreadMore--;
						//if (countAverageBoxesPerThreadMore == 0)
					}

					if((countAverageBoxesPerThreadMore == 0) || workLen[i] == averageBoxesPerThread + 1) 
					{
						curThreadWeTakeBoxesIndex = j;
						break;	
					}
				}
						
			}
			
		}
		if(countAverageBoxesPerThreadMore == 0) {
			stopMore = i;
			break;
		}
		
	}
	
	/*
	curThreadWeTakeBoxesIndex = 0;
	for(int i = 0; i < n; i++)
	{

		if(workLen[i] < averageBoxesPerThread)
		{
			
			for(int j = curThreadWeTakeBoxesIndex; j < n; j++)
			{
				//if(workLen[j] == averageBoxesPerThread + 1 && i > j 
				if(workLen[j] > averageBoxesPerThread)  
				{
					plusOne = 0;
					numBoxesWeTake = (averageBoxesPerThread+plusOne) - workLen[i] <= workLen[j] - (averageBoxesPerThread) ? (averageBoxesPerThread+plusOne) - workLen[i] : workLen[j] - (averageBoxesPerThread);
					workLen[j] -= numBoxesWeTake;
					//memcpy(boxes + i*m*(2*dim+3) + (workLen[i])*(2*dim+3), boxes + j*m*(2*dim+3) + (workLen[j])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
					workLen[i] += numBoxesWeTake;	
					if(workLen[i] == averageBoxesPerThread) 
					{
						curThreadWeTakeBoxesIndex = j;
						break;	
					}
				}
						
			}
			
		}
		
	}
	*/

	printf("\n\n");
	for(int i = 0; i < n; i++)
	{		
		printf("%d\t", workLen[i]);	
	}
	printf("\n\n");
				
}


BalancingInfo balancingOnCPU_v2(double* boxes, int *workLen, int n, int m, int dim)
{
	printf("\n\n");
	for(int i = 0; i < n; i++)
	{		
		printf("%d\t", workLen[i]);	
	}
	printf("\n\n");
				
				
	
	int numWorkBoxes;
	int averageBoxesPerThread;
	int curThreadWeTakeBoxesIndex;
	int curThreadWeTakeBoxesCount;
	int numBoxesWeTake;
	int boxIndex;
	int countAverageBoxesPerThreadMore;
	int plusOne;
	int i,j;
	int countMemoryCopies;
	
	BalancingInfo balancingInfo;
	balancingInfo.numThreads = n;
	balancingInfo.maxNumberOfBoxesPerThread = m;
	balancingInfo.version = BalancinVersion.WITHOUT_SORT_ON_CPU;
	
	auto start = std::chrono::high_resolution_clock::now();

	for (i = 0; i < n; i++) {
		numWorkBoxes += workLen[i]; 	
	}
	averageBoxesPerThread = numWorkBoxes / n;		
	if(averageBoxesPerThread == 0) averageBoxesPerThread = averageBoxesPerThread + 1;
			
	curThreadWeTakeBoxesIndex = 0;
	countMemoryCopies = 0;
	for (i = 0; i < n; i++) {
		if (workLen[i] < averageBoxesPerThread) {
			for (j = curThreadWeTakeBoxesIndex; j < n; j++) {
				if (workLen[j] > averageBoxesPerThread) {		
					numBoxesWeTake = averageBoxesPerThread - workLen[i] <= workLen[j] - averageBoxesPerThread ? averageBoxesPerThread - workLen[i] : workLen[j] - averageBoxesPerThread;
					workLen[j] -= numBoxesWeTake;
					//memcpy(inBox + (i+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[i])*(2*inRank+3), inBox + (j+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*numBoxesWeTake);
					workLen[i] += numBoxesWeTake;	
					countMemoryCopies++;
					if (workLen[i] == averageBoxesPerThread) break;	
				}			
			}
			curThreadWeTakeBoxesIndex = j;
		}
				
	}
			
			
	boxIndex = 0;
	for (i = 0; i < n; i++) {
		if (workLen[i] == averageBoxesPerThread) {
			for (j = curThreadWeTakeBoxesIndex; j < n; j++) {
				if (workLen[j] > averageBoxesPerThread + 1) {
					numBoxesWeTake = 1;
					workLen[j] -= numBoxesWeTake;
					//memcpy(inBox + (i+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[i])*(2*inRank+3), inBox + (j+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*numBoxesWeTake);
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
	
	balancingInfo.time = (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
	
	balancingInfo.numberOfMemoryCopies = countMemoryCopies;
	
	printf("\n\n");
	for (i = 0; i < n; i++) {		
		printf("%d\t", workLen[i]);	
	}
	printf("\n\n");
	printf("countMemoryCopies: %d\n", countMemoryCopies);
	
	return balancingInfo
				
}



BalancingInfo balancingOnCPU_v3(double* boxes, int *workLen, int n, int m, int dim)
{

	printf("\n\n");
	for (int i = 0; i < n; i++) {		
		printf("%d\t", workLen[i]);	
	}
	printf("\n\n");
	
	int *workLenIndexes = new int[n];
	for (int i = 0; i < n; i++) {
		workLenIndexes[i] = i;
	}
	

	int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = -1;
	int curThreadWeTakeBoxesCount = 0;
	int numBoxesWeTake = 0;
	int countAverageBoxesPerThreadMore = 0;
	int curThreadWeGiveBoxesIndex = -1;
	int plusOne = 0;
	int i,j;
	int countMemoryCopies = 0;
	
	BalancingInfo balancingInfo;
	balancingInfo.numThreads = n;
	balancingInfo.maxNumberOfBoxesPerThread = m;
	balancingInfo.version = BalancinVersion.WITH_SORT_ON_CPU;
	
	
	auto start = std::chrono::high_resolution_clock::now();

	for (i = 0; i < n; i++) {
		numWorkBoxes += workLen[i]; 	
	}
	averageBoxesPerThread = numWorkBoxes / n;
			
	averageBoxesPerThread = numWorkBoxes / n;		
	countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*n;
	
	curThreadWeTakeBoxesIndex = n-1;
	curThreadWeGiveBoxesIndex = 0;
			
	sortQuickRecursive(workLenIndexes,workLen,n);
	printf("\nStart\n");
	
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
		//memcpy(inBox + (i+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[i])*(2*inRank+3), inBox + (j+blockIdx.x * BLOCK_SIZE)*SIZE_BUFFER_PER_THREAD*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*numBoxesWeTake);
		workLen[curThreadWeGiveBoxesIndex] += numBoxesWeTake;
		countMemoryCopies++;
			
	}
		

	auto end = std::chrono::high_resolution_clock::now();
	
	balancingInfo.time = (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
	
	balancingInfo.numberOfMemoryCopies = countMemoryCopies;

	printf("\n\n");
	for (int i = 0; i < n; i++) {		
		printf("%d\t", workLen[i]);	
	}
	printf("\n\n");
	printf("countMemoryCopies: %d\n", countMemoryCopies);
	
	return balancingInfo;
				
}
