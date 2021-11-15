#pragma once
	
#define MASTER 0				// define master rank
// input and output file name
#define INPUT_FILE_NAME "C:\\Users\\cudauser\\Desktop\\ParallelFinalProject_ID312672942\\MPI+OMP+CUDA\\data.txt" 
#define OUTPUT_FILE_NAME "C:\\Users\\cudauser\\Desktop\\ParallelFinalProject_ID312672942\\MPI+OMP+CUDA\\output.txt"
// assignment requirements
#define MAX_DIM 20				// max number of dimantion
#define MIN_SIZE_POINTS 100000	// min number of points allowed
#define MAX_SIZE_POINTS 500000	// max number of points allowed	
#define MAX_LIMIT 100			// calculation time limit
#define NUM_OF_THREADS 1000		// num of threads

//include
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "mpi.h"

// point struct
struct Point
{
	double values[MAX_DIM];		// coardinats of point
	double velocity[MAX_DIM];	// velocity vector coardinats
	int groupIdentifier;		// classification to A or B
	int k;						// dimantion
}typedef point_t;

// output struct - includes the result of every process
struct Output
{
	double w[MAX_DIM];			// weight vector
	double alpha;				// alpha parameter that was read from data
	double q;					// quality of the classification
	double t;					// minimal time that was reach to succes , initialized with limit value
	int k;						// dimantion
}typedef output_t;

//mpi methods
void createOutputDataType(MPI_Datatype* OutputMPIType);
void createPointDataType(MPI_Datatype* PointMPIType);
void sentPointsAndRangeByT(int n, int numprocs, point_t* pointsArr, int* tRange, MPI_Datatype PointMPIType, double dt, double tMax);

//file mrthods
point_t* getInputFromFile(char* fileName, int* n, int* k, double* dt, double* tMax, double* alpha, int* limit, double* qc);
void saveToOutputFile(output_t o);

// print methods
void printPoint(const point_t point);
void printPointsArr(int n, point_t* pointsArr);
void printOutput(const output_t o);
void printOutputArr(int numprocs, output_t* oArray);

// find min t from the output result that was returned from the procceses
int findTMin(int numrocs, output_t* oArray);

// free allocation
void freeAll(int* array, point_t* pointsArray, cudaError_t cudaStatus);

// CUDA methods
//cudaError_t Save_Array_Points_And_Weights(point_t *pointsArr, int n, double *weights, int k);
//cudaError_t  copy_w(double *weights, int k);
cudaError_t free_All(void);
//cudaError_t Calculate(int n, int k, double *weights, int *arr);







