/*
 * CPUGlobOptBFSWithMmapAndOMP.cpp
 *
 *  Created on: 10 Aug 2017
 *  Author: Leonid Tkachenko
 */


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


#include "iGlobOpt.h"



/**
*	Calculus minimum value for function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param inNumBoxesSplitCoeff number of parts for each dimension
*	@param inEps required accuracy
*	@param inMaxIter maximum count of iterations
*	@param inFun pointer to optimazing function
*	@param outBox pointer to optimal box
*	@param outMin pointer to optimal value
*	@param outEps pointer to reached accuracy
*	@param outEps pointer to status of solving optimization problem
*/


void calcOptValueOnCPUBFSWithMmapAndOMP(const double *_boxes, int _numBoxes, int _rank, int _splitCoeff, void (*_fun)(const double *, int, double *), double _eps, double *_min, GlobOptErrors *_status, double *_argmin)
{

	int numThreads = omp_get_num_threads();
	double *workBoxes = new double[_rank*MAX_BOXES_IN_BUFFER*2];
	double *restBoxesToSplit = new double[_rank*MAX_BOXES_IN_BUFFER*2];
	double *funBounds = new double[ARRAY_BOUNDS_LENGTH*MAX_BOXES_IN_BUFFER];


	long long wc = 0;
	
	//copy Input Boxes in work set #1
	try
	{
		memcpy(restBoxesToSplit, _boxes, _rank*2*_numBoxes*sizeof(double));
	}
	catch (std::exception &e)
	{
		delete [] workBoxes;
		delete [] restBoxesToSplit;
		delete [] funBounds;
		std::cerr << "Error coping input boxes in workBuffer: " << e.what() << std::endl;
	}

	//calculate initial function record
	try
	{
		_fun(_boxes,_rank,funBounds);
	}
	catch(std::exception &e)
	{
		std::cerr << "Error computing initial function record: " << e.what() << std::endl;
	}
	double funRecord = funBounds[GO_POSITION_FUN_RECORD];
	double funLB;


	funRecord = -39.1661657038*_rank;
	
	
	int numWorkBoxes = _numBoxes;

	int fd = open(FILEPATH,O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
	if(fd == -1)
	{
		perror("Error opening file for writing");
		exit(EXIT_FAILURE);

	}

	int result = lseek(fd, SIZE_BUFFER_FILE,SEEK_SET);
	if(result == -1)
	{
		close(fd);
		perror("Error calling lseel");
		exit(EXIT_FAILURE);
	}

	result = write(fd,"",1);
	if(result != 1)
	{
		close(fd);
		perror("Error calling write");
		exit(EXIT_FAILURE);
	}


	int numBoxesInFile = 0;
	int s;
	double *map;
	off_t pa_offset,offset;
	
	
	double h1 = _boxes[1] - _boxes[0];
	double hInd1 = 0;
	
	for(int i = 0; i < _rank; i++)
	{
		if(h1 < _boxes[i*_rank + 1] - _boxes[i*_rank])
		{
			h1 = _boxes[i*_rank + 1] - _boxes[i*_rank];
			hInd1 = i;
		}
	}
	
	for(int n = 0; n < numWorkBoxes*1024; n++)
	{
		for(int i = 0; i < _rank; i++)
		{
			if(i == hInd1)
			{
				restBoxesToSplit[n*2*_rank + i*2] = _boxes[i*2] + h1/1024.0*n;
				restBoxesToSplit[n*2*_rank + i*2 + 1] = _boxes[i*2] + h1/1024.0*(n+1);
			}
			else
			{
				restBoxesToSplit[n*2*_rank + i*2] = _boxes[i*2];
				restBoxesToSplit[n*2*_rank + i*2 + 1] = _boxes[i*2 + 1];
			}
		}

	}
	
	numWorkBoxes = 1024;
	
	
	auto start = std::chrono::high_resolution_clock::now();

	char tp[100];
	//While global optimum not found
	while(true)
	{
		//scanf("%s",tp);
		//std::cin.get();
		std::cout << "\n\n-------------------------";
		std::cout << "wc:  " << wc << "\n\n";
		
		/*
		for(int i = 0; i < numWorkBoxes; i++)
		{
			for(int j = 0; j < _rank; j++)
			{
				std::cout << "[" << restBoxesToSplit[(i*_rank+j)*2] << ";" <<  restBoxesToSplit[(i*_rank+j)*2+1] << "]\t";
			}
			std::cout << ",";
			
		}
		*/

		//Workin with file
		if(numWorkBoxes*_splitCoeff >= MAX_BOXES_IN_BUFFER)
		{

			s = numWorkBoxes/PART_BUFFER_TO_FILE;
			offset = numBoxesInFile*_rank*2*sizeof(double);
			pa_offset = offset & ~(sysconf(_SC_PAGE_SIZE) - 1);
			map = (double *)mmap(0,s*_rank*2*sizeof(double),PROT_READ | PROT_WRITE, MAP_SHARED, fd, pa_offset);
			if(map == MAP_FAILED)
			{
				close(fd);
				delete [] restBoxesToSplit;
				delete [] workBoxes;
				delete [] funBounds;
				perror("MAP FAILED");
				exit(EXIT_FAILURE);
			}
			numBoxesInFile += s;
			memcpy(map,restBoxesToSplit+(numWorkBoxes - s)*_rank*2,s*_rank*2*sizeof(double));
			if(munmap(map,s*_rank*2*sizeof(double)) == -1)
			{
				close(fd);
				delete [] restBoxesToSplit;
				delete [] workBoxes;
				delete [] funBounds;
				perror("Error un-mapping the file");
				exit(EXIT_FAILURE);
			}
			numWorkBoxes -= s;

		}
		else if(numWorkBoxes*_splitCoeff <= MAX_BOXES_IN_BUFFER/PART_BUFFER_FROM_FILE && numBoxesInFile > 0)
		{
			s = MAX_BOXES_IN_BUFFER/PART_BUFFER_FROM_FILE;
			if(numBoxesInFile <= s)  s = numBoxesInFile;

			offset = numBoxesInFile-s > 0? numBoxesInFile-s : 0;

			offset = offset*_rank*2*sizeof(double);
			pa_offset = offset & ~(sysconf(_SC_PAGE_SIZE) - 1);

			map = (double *)mmap(0,s*_rank*2*sizeof(double),PROT_READ | PROT_WRITE, MAP_SHARED, fd, pa_offset);
			if(map == MAP_FAILED)
			{
				close(fd);
				delete [] restBoxesToSplit;
				delete [] workBoxes;
				delete [] funBounds;
				perror("MAP FAILED");
				exit(EXIT_FAILURE);
			}
			numBoxesInFile -= s;
			memcpy(restBoxesToSplit+numWorkBoxes*_rank*2,map,s*_rank*2*sizeof(double));
			if(munmap(map,s*_rank*2*sizeof(double)) == -1)
			{
				perror("Error un-mapping the file");
				exit(EXIT_FAILURE);
			}
			numWorkBoxes += s;
		}
#pragma omp parallel for
		//Splitting all work Boxes
		for(int k = 0; k < numWorkBoxes; k++)
		{
			//Searching max dimension to split
			int maxDimensionIndex = 0;
			double maxDimension = restBoxesToSplit[(k*_rank)*2 + 1] - restBoxesToSplit[(k*_rank)*2];
			double h; //?ToDO : is it correct to declare variables here
			for(int i = 0; i < _rank; i++)
			{
				h = (restBoxesToSplit[(k*_rank+i)*2 + 1] - restBoxesToSplit[(k*_rank+i)*2]);
				if (maxDimension < h)
				{
					maxDimension = h;
					maxDimensionIndex = i;
				}

			}
			h = maxDimension/_splitCoeff;

			for(int n = 0; n < _splitCoeff; n++)
			{
				for(int i = 0; i < _rank; i++)
				{
					if (i==maxDimensionIndex)
					{
						workBoxes[((k*_splitCoeff + n)*_rank+i)*2] = restBoxesToSplit[(k*_rank+i)*2] + h*n;
						workBoxes[((k*_splitCoeff + n)*_rank+i)*2 + 1] = restBoxesToSplit[(k*_rank+i)*2] + h*(n+1);
					} else
					{
						workBoxes[((k*_splitCoeff + n)*_rank+i)*2] = restBoxesToSplit[(k*_rank+i)*2];
						workBoxes[((k*_splitCoeff + n)*_rank+i)*2 + 1] = restBoxesToSplit[(k*_rank+i)*2 + 1];
					}
				}
				
				_fun(&workBoxes[((k*_splitCoeff + n)*_rank)*2],_rank,&funBounds[(k*_splitCoeff + n)*ARRAY_BOUNDS_LENGTH]);
			}
		}

		funLB = funBounds[GO_POSITION_LB];
#pragma omp parallel for reduction(min: funRecord) reduction(min: funLB)
		for(int i = 0; i < numWorkBoxes*_splitCoeff; i++)
		{
			 if(funRecord > funBounds[i*ARRAY_BOUNDS_LENGTH + GO_POSITION_FUN_RECORD] )
			{
				funRecord = funBounds[i*ARRAY_BOUNDS_LENGTH+GO_POSITION_FUN_RECORD];
			}

			if (funLB > funBounds[i*ARRAY_BOUNDS_LENGTH + GO_POSITION_LB]) funLB = funBounds[i*ARRAY_BOUNDS_LENGTH + GO_POSITION_LB];
		}

		//checking if the global minimum is found
		double curEps = funRecord - funLB < 0 ? -(funRecord - funLB) : funRecord - funLB;
		/*
		if(curEps < _eps  && numBoxesInFile == 0)
		{
			*_min = funRecord;
			*_status = GO_SUCCESS;
			delete [] restBoxesToSplit;
			delete [] workBoxes;
			delete [] funBounds;
			close(fd);
			
			return;
		}
		*/

		//Saving appropriate boxes to split
		int cnt = 0;
		for(int i = 0; i < numWorkBoxes*_splitCoeff; i++)
		{
			if(funBounds[i*ARRAY_BOUNDS_LENGTH + GO_POSITION_LB] <= funRecord - _eps)
			{
				for(int j = 0; j < _rank; j++)
				{
					restBoxesToSplit[(cnt*_rank+j)*2] = workBoxes[(i*_rank+j)*2];
					restBoxesToSplit[(cnt*_rank+j)*2+1] = workBoxes[(i*_rank+j)*2+1];
				}
				cnt++;
			}
			else 
			{
				wc++;
				for(int j = 0; j < _rank; j++)
				{
					printf("[%f; %f]\t",workBoxes[(i*_rank+j)*2],workBoxes[(i*_rank+j)*2+1]);
				}
				printf("%f\t%f\t%f\n",funBounds[i*ARRAY_BOUNDS_LENGTH + GO_POSITION_RB],funBounds[i*ARRAY_BOUNDS_LENGTH + GO_POSITION_LB],funBounds[i*ARRAY_BOUNDS_LENGTH + GO_POSITION_FUN_RECORD]);
				
			}
				
		}
		numWorkBoxes = cnt;
		
		//std::cout << "min = ";
		//printf("%.7f",funRecord);
		//std::cout << "\tfunLb = " << funLB << "\n";
		
		//std::cout << "\twc = " << wc << "\n";

		if(numWorkBoxes == 0 && numBoxesInFile == 0)
		{
			*_status = GO_WORKBUFFER_IS_EMPTY;
			delete [] restBoxesToSplit;
			delete [] workBoxes;
			delete [] funBounds;
			close(fd);
			auto end = std::chrono::high_resolution_clock::now();
			std::cout << "time in millisecs: " << ((std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count())/1 << "\t";
			std::cout << "\twc = " << wc << "\n";
			
			return;
		}

	}
}





