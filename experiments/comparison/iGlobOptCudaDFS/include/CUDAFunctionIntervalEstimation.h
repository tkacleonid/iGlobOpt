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
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
__device__ void fnCalcFunLimitsStyblinski_CUDA_v1(double *inBox, int inRank)
{
	double sup = 0;
	double sub = 0;
	double sup1,sub1,sup2,sub2,val = 0,var1,var2,var3,x;
	int i;

	for(i = 0; i < inRank; i++)
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
	

	inBox[2*inRank + 0] = sub;
	inBox[2*inRank + 1] = sup;
	inBox[2*inRank+2] = val;
	
}



