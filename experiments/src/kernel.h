
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
	
	double *boxes = new double[1*n*m*dim];
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
	
}



