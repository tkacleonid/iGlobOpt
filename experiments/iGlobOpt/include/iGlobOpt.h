/*
 * iGlobOpt.h
 *
 *  Created on: 10 Aug 2017
 *      Author: Leonid Tkachenko
 */

#ifndef INCLUDE_IGLOBOPT_H_
#define INCLUDE_IGLOBOPT_H_


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <chrono>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <exception>
#include <time.h>
#include <cstdio>


#define FILEPATH "mmaped.bin"


const std::string MMAP_FILEPATH = "mmaped.bin";
const long long MAX_BOXES_IN_BUFFER = 100000000;
const int ARRAY_BOUNDS_LENGTH = 3;
const long long SIZE_BUFFER_FILE = 2000000000L;
const int PART_BUFFER_TO_FILE = 2;
const int PART_BUFFER_FROM_FILE = 4;


//Poisitions in computing box
enum PositionBounds
{
	GO_POSITION_LB = 0,
	GO_POSITION_RB = 1,
	GO_POSITION_FUN_RECORD = 2
};

//ToDO : For future use. Operation with errors
enum GlobOptErrors
{
	GO_WORKBUFFER_IS_FULL,
	GO_WORKBUFFER_IS_EMPTY,
	GO_SUCCESS
};


void calcOptValueOnCPUBFS(const double *_boxes, int _numBoxes,int _dim, int _splitCoeff, void (*_fun)(const double *, int, double *), double _eps, double *_min, GlobOptErrors *_status, double *_argmin);

void calcOptValueOnCPUBFSWithMmap(const double *_boxes, int _numBoxes,int _dim, int _splitCoeff, void (*_fun)(const double *, int, double *), double _eps, double *_min, GlobOptErrors *_status, double *_argmin);

void calcOptValueOnCPUBFSWithOMP(const double *_boxes, int _numBoxes,int _dim, int _splitCoeff, void (*_fun)(const double *, int, double *), double _eps, double *_min, GlobOptErrors *_status, double *_argmin);

void calcOptValueOnCPUBFSWithMmapAndOMP(const double *_boxes, int _numBoxes, int _dim, int _splitCoeff, void (*_fun)(const double *, int, double *), double _eps, double *_min, GlobOptErrors *_status, double *_argmin, double _initFunRecord);






#endif /* INCLUDE_IGLOBOPT_H_ */
