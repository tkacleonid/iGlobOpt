
#include "CUDAGlobalOptimization.h"
#include "interval.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <exception>
#include <stdlib.h>

void balancingOnCPU(int n, int m, int dim);

int main()
{	
	
	srand(time(NULL));
	
	balancingOnCPU(100, 100, 2);
		
	return 0;
}


void balancingOnCPU(int n, int m, int dim)
{
	//Initialize random seed
	srand(time(NULL));
	
	double *boxes = new double[(2*dim+3) * n*m];
	int *workLen = new int[n];
	
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

		if(workLen[i] == averageBoxesPerThread + 1 && countAverageBoxesPerThreadMore > 0)
		{
			countAverageBoxesPerThreadMore--;
		}
		else if(workLen[i] < averageBoxesPerThread || (workLen[i] == averageBoxesPerThread && countAverageBoxesPerThreadMore > 0))
		{
			for(int j = curThreadWeTakeBoxesIndex; j < n; j++)
			{
				if(workLen[j] > averageBoxesPerThread)
				{
					if	(countAverageBoxesPerThreadMore > 0) plusOne = 1;
					else plusOne = 0;
					numBoxesWeTake = (averageBoxesPerThread+plusOne) - workLen[i] <= workLen[j] - (averageBoxesPerThread) ? (averageBoxesPerThread+plusOne) - workLen[i] : workLen[j] - (averageBoxesPerThread);
					if(numBoxesWeTake + workLen[i] == averageBoxesPerThread + 1)  countAverageBoxesPerThreadMore--;
					workLen[j] -= numBoxesWeTake;
					//memcpy(boxes + i*m*(2*dim+3) + (workLen[i])*(2*dim+3), boxes + j*m*(2*dim+3) + (workLen[j])*(2*dim+3), sizeof(double)*(2*dim+3)*numBoxesWeTake);
					workLen[i] += numBoxesWeTake;	
					if((workLen[i] == averageBoxesPerThread && countAverageBoxesPerThreadMore == 0) || workLen[i] == averageBoxesPerThread + 1) 
					{
						curThreadWeTakeBoxesIndex = j;
						break;	
					}
				}
						
			}
			
		}		
	}

	printf("\n\n");
	for(int i = 0; i < n; i++)
	{		
		printf("%d\t", workLen[i]);	
	}
	printf("\n\n");
				
}



