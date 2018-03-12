
#include "CUDAGlobalOptimization.h"
#include "interval.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <exception>
#include <stdlib.h>


int main()
{	
	
	srand(time(NULL));
	
	double *boxes = new double[2*n*m*dim];
	int *numberOfBoxes = new int[n];
	
	for(int i = 0; i < n; i++)
	{
		numberOfBoxes[i] = rand()%(m+1);
		for(int j = 0; j < numberOfBoxes[i]; j++)
		{
			for(int k = 0; k < dim; k++)
			{
				boxes[2*i*m*dim + 2*j*dim] = (rand()%(m+1))/(double) n;
				boxes[2*i*m*dim + 2*j*dim + 1] = (rand()%(m+1))/(double) n;
			}
		}
		
	}
		
	return 0;
}


void balancingOnCPU(int n, int m, int dim)
{
	//Initialize random seed
	srand(time(NULL));
	
	double *boxes = new double[2*n*m*dim];
	int *workLen = new int[n];
	
	for(int i = 0; i < n; i++)
	{
		workLen[i] = rand()%(m+1);
		for(int j = 0; j < workLen[i]; j++)
		{
			for(int k = 0; k < dim; k++)
			{
				boxes[2*i*m*dim + 2*j*dim] = (rand() % (m+1))/(double) n;
				boxes[2*i*m*dim + 2*j*dim + 1] = (rand() % (m+1))/(double) n;
			}
		}		
	}
	
	int numWorkBoxes = 0;
	int averageBoxesPerThread = 0;
	int curThreadWeTakeBoxesIndex = -1;
	int curThreadWeTakeBoxesCount = 0;
	int numBoxesWeTake = 0;
	int boxIndex = 0;
	int countAverageBoxesPerThreadMore = 0;
	
	

	for(i = 0; i < n; i++)
	{
		numWorkBoxes += workLen[i]; 	
	}
	
	averageBoxesPerThread = numWorkBoxes / m;
			
	if (averageBoxesPerThread == 0) averageBoxesPerThread = averageBoxesPerThread + 1;
	countAverageBoxesPerThreadMore = numWorkBoxes - averageBoxesPerThread*m;
	
	
	
	
			
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



