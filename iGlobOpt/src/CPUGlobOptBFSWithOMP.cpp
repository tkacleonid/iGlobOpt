/*
 * CPUGlobOptBFS.cpp
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


void calcOptValueOnCPUBFSWithOMP(const double *_boxes, int _numBoxes, int _rank, int _splitCoeff, void (*_fun)(const double *, int, double *), double _eps, double *_min, GlobOptErrors *_status, double *_argmin)
{

	int numThreads = omp_get_num_threads();
	double *workBoxes = new double[_rank*MAX_BOXES_IN_BUFFER*2];
	double *restBoxesToSplit = new double[_rank*MAX_BOXES_IN_BUFFER*2];
	double *funBounds = new double[ARRAY_BOUNDS_LENGTH*MAX_BOXES_IN_BUFFER];


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


	int numWorkBoxes = _numBoxes;

	//While global optimum not found
	while(true)
	{

#pragma omp parallel  for num_threads(numThreads)
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
#pragma omp parallel  for num_threads(numThreads) reduction(min: funRecord, min: funLB)
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
		if(curEps < _eps)
		{
			*_min = funRecord;
			*_status = GO_SUCCESS;
			delete [] restBoxesToSplit;
			delete [] workBoxes;
			delete [] funBounds;
			return;
		}

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
		}
		numWorkBoxes = cnt;

		if(numWorkBoxes == 0)
		{
			*_status = GO_WORKBUFFER_IS_EMPTY;
			delete [] restBoxesToSplit;
			delete [] workBoxes;
			delete [] funBounds;
			return;
		}

		if(numWorkBoxes*_splitCoeff > MAX_BOXES_IN_BUFFER)
		{
			*_status = GO_WORKBUFFER_IS_FULL;
			delete [] restBoxesToSplit;
			delete [] workBoxes;
			delete [] funBounds;
			return;
		}

	}
}





