#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_device_runtime_api.h>

//#include <math_functions_dbl_ptx3.h>
#include "interval.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>



void balancingOnCPU(double* boxes, int *workLen,int n, int m, int dim);
void balancingOnCPU2(int n, int m, int dim);
void sortQuickRecursive(int *indexes,int *ar,  const int n);
void quickSortBase(int *indexes,int *ar, const int l, const int r);
void balancingOnCPU_v3(double* boxes, int *workLen, int n, int m, int dim);
void initializeBoxes(double* boxes, int *workLen, int n, int m, int dim);