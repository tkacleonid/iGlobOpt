#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_device_runtime_api.h>

#include "interval.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>


/**
*	Calculus Interval for Styblinski function on CUDA
*	@param inbox pointer to Box
*	@param inDim number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsStyblinski_CUDA_v1(double *inBox, int inDim)
{
	double sup = 0;
	double sub = 0;
	double sup1,sub1,sup2,sub2,val = 0,var1,var2,var3,x;
	int i;

	for(i = 0; i < inDim; i++)
	{
			
		var1 = inBox[i*2 + 1]*inBox[i*2 + 1];
		var2 = inBox[i*2 + 1]*inBox[i*2];
		var3 = inBox[i*2]*inBox[i*2];
		
		
		
		
		sub1 = fmin(fmin(var1,var2),var3);
		sup1 = fmax(fmax(var1,var2),var3);
		
		var1 = sub1*sub1;
		var2 = sub1*sup1;
		var3 = sup1*sup1;
		
		sub2 = fmin(fmin(var1,var2),var3);
		sup2 = fmax(fmax(var1,var2),var3);

		sub += (sub2 - 16*sup1 + 5*fmin(inBox[i*2 + 1],inBox[i*2]))/2.0;
		sup += (sup2 - 16*sub1 + 5*fmax(inBox[i*2 + 1],inBox[i*2]))/2.0;
		
		
		
		

		//sub += sub1;
		//sup += sup1;

		x = (inBox[i*2 + 1] + inBox[i*2])/2;
		val += (x*x*x*x - 16*x*x + 5*x)/2.0;
			
	}
	

	inBox[2*inDim + 0] = sub;
	inBox[2*inDim + 1] = sup;
	inBox[2*inDim+2] = val;
	
}


/**
*	Calculus Interval for Multiple function on GPU
*	@param inbox pointer to Box
*	@param inDim number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsMultiple2_CUDA(double *inBox, int inDim, double *outLimits)
{
	
	double x1 = (inBox[0]+inBox[1])/2;
	double x2 = (inBox[2]+inBox[3])/2;

	double var1 = inBox[0]*inBox[2];
	double var2 = inBox[0]*inBox[3];
	double var3 = inBox[1]*inBox[2];
	double var4 = inBox[1]*inBox[3];

	outLimits[0] = fmin(fmin(var1,var2),fmin(var3,var4));
	outLimits[1] = fmax(fmax(var1,var2),fmax(var3,var4));
	outLimits[2] = x1*x2;
}

/**
*	Calculus Interval for Hyperbolic function on GPU
*	@param inbox pointer to Box
*	@param inDim number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsHypebolic2_CUDA(double *inBox, int inDim, double *outLimits)
{
	double limits[2];
	double limits2[2];

	double x1 = (inBox[0]+inBox[1])/2;
	double x2 = (inBox[2]+inBox[3])/2;

	double var1 = inBox[0]*inBox[0];
	double var2 = inBox[0]*inBox[1];
	double var3 = inBox[1]*inBox[1];

	limits[0] = var2 < 0 ? 0 : fmin(fmin(var1,var2),var3);
	limits[1] = fmax(fmax(var1,var2),var3);

	var1 = inBox[2]*inBox[2];
	var2 = inBox[2]*inBox[3];
	var3 = inBox[3]*inBox[3];

	limits2[0] = var2 < 0 ? 0 : fmin(fmin(var1,var2),var3);
	limits2[1] = fmax(fmax(var1,var2),var3);

	outLimits[0] = limits[0] - limits2[1];
	outLimits[1] = limits[1] - limits2[0];
	outLimits[2] = x1*x1-x2*x2;
}

/**
*	Calculus Interval for AluffiPentini function
*	@param inbox pointer to Box
*	@param inDim number of variables
*	@param outlimits pointer to estimated function limit
*/

__device__ void fnCalcFunLimitsAluffiPentini2_CUDA(double *inBox, int inDim, double *outLimits)
{

	double limits[2];
	double limits2[2];

	double x1 = (inBox[0]+inBox[1])/2;
	double x2 = (inBox[2]+inBox[3])/2;

	double var1 = inBox[0]*inBox[0];
	double var2 = inBox[0]*inBox[1];
	double var3 = inBox[1]*inBox[1];

	limits[0] = var2 < 0 ? 0 : fmin(fmin(var1,var2),var3);
	limits[1] = fmax(fmax(var1,var2),var3);

	var1 = inBox[2]*inBox[2];
	var2 = inBox[2]*inBox[3];
	var3 = inBox[3]*inBox[3];

	limits2[0] = limits[0]*limits[0];
	limits2[1] = limits[1]*limits[1];

	outLimits[0] = 0.25*limits2[0] - 0.5*limits[1] + 0.1*inBox[0] + 0.5*(var2 < 0 ? 0 : fmin(fmin(var1,var2),var3));
	outLimits[1] = 0.25*limits2[1] - 0.5*limits[0] + 0.1*inBox[1] + 0.5*(fmax(fmax(var1,var2),var3));
	outLimits[2] = 0.25*pow(x1,4.0)-0.5*pow(x1,2.0) + 0.1*x1 + 0.5*pow(x2,2.0);
}

/**
*	Calculus Interval for Rozenbroke function
*	@param inbox pointer to Box
*	@param inDim number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsRozenbroke_CUDA(double *inBox, int inDim)
{
	double sup = 0;
	double sub = 0;
	double sup1,sub1,sup2,sub2,a,b,val = 0,var1,var2,var3,x1,x2;
	int i;

	for(i = 0; i < inDim - 1; i++)
	{
		sub1 = 1 - inBox[i*2 + 1];
		sup1 = 1 - inBox[i*2];
		
		var1 = sup1*sup1;
		var2 = sup1*sub1;
		var3 = sub1*sub1;
		
		sub1 = (sub1*sup1 < 0) ? 0 : fmin(fmin(var1,var2),var3);
		sup1 = fmax(fmax(var1,var2),var3);	

		var1 = inBox[i*2 + 1]*inBox[i*2 + 1];
		var2 = inBox[i*2 + 1]*inBox[i*2];
		var3 = inBox[i*2]*inBox[i*2];

		a = (inBox[i*2 + 1]*inBox[i*2] < 0) ? 0 : fmin(fmin(var1,var2),var3);
		b = fmax(fmax(var1,var2),var3);
		
		sub2 = inBox[(i+1)*2] - b;
		sup2 = inBox[(i+1)*2 + 1] - a;
		
		var1 = sup2*sup2;
		var2 = sup2*sub2;
		var3 = sub2*sub2;
		
		sub2 = (sub2*sup2 < 0) ? 0 : 100*fmin(fmin(var1,var2),var3);
		sup2 = 100*fmax(fmax(var1,var2),var3);

		sub += sub1 + sub2;
		sup += sup1 + sup2;

		x1 = (inBox[i*2 + 1] + inBox[i*2])/2;
		x2 = (inBox[(i+1)*2 + 1] + inBox[(i+1)*2])/2;
		val += ((1 - x1)*(1 - x1) + 100*(x2-x1*x1)*(x2-x1*x1));
	}

	
	inBox[inDim*2] = sub;
	inBox[inDim*2 + 1] = sup;
	inBox[inDim*2 + 2] = val;
	
}


/**
*	Calculus Interval for Rozenbroke function
*	@param inbox pointer to Box
*	@param inDim number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsStyblinski_CUDA(double *inBox, int inDim)
{
	double sup = 0;
	double sub = 0;
	double sup1,sub1,sup2,sub2,val = 0,var1,var2,var3,x;
	int i;

	for(i = 0; i < inDim; i++)
	{
			
		var1 = inBox[i*2 + 1]*inBox[i*2 + 1];
		var2 = inBox[i*2 + 1]*inBox[i*2];
		var3 = inBox[i*2]*inBox[i*2];
		
		
		
		
		sub1 = fmin(fmin(var1,var2),var3);
		sup1 = fmax(fmax(var1,var2),var3);
		
		var1 = sub1*sub1;
		var2 = sub1*sup1;
		var3 = sup1*sup1;
		
		sub2 = fmin(fmin(var1,var2),var3);
		sup2 = fmax(fmax(var1,var2),var3);

		sub += (sub2 - 16*sup1 + 5*fmin(inBox[i*2 + 1],inBox[i*2]))/2.0;
		sup += (sup2 - 16*sub1 + 5*fmax(inBox[i*2 + 1],inBox[i*2]))/2.0;
		
		
		
		

		//sub += sub1;
		//sup += sup1;

		x = (inBox[i*2 + 1] + inBox[i*2])/2;
		val += (x*x*x*x - 16*x*x + 5*x)/2.0;
			
	}
	

	inBox[2*inDim + 0] = sub;
	inBox[2*inDim + 1] = sup;
	inBox[2*inDim+2] = val;
	
}



