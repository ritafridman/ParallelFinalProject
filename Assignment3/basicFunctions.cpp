#define _CRT_SECURE_NO_WARNINGS
#include "basicFunctions.h"

// print single point
void printPoint(const point_t point) {
	int i, j;
	for (i = 0; i < point.k - 1; i++) {			// loop of point dimantion
		printf("%f ", (point.values[i]));		// print point values
	}
	for (j = 0; j < point.k - j; j++) {			// loop of point dimantion
		printf("%f ", (point.velocity[j]));		//print velocity vector
	}
	printf("%d\n", point.groupIdentifier);		//print point classification
	fflush(stdout);								// clean
}

//print array of points
void printPointsArr(int n, point_t* pointsArr) {
	int j;
	for (j = 0; j < n; j++)						// loop of num of points
		printPoint(pointsArr[j]);				// print single point
	fflush(stdout);								// clean
}

//print output result
void printOutput(const output_t o) {
	int i;
	printf("Alpha minimum = %f \n", o.alpha);	// alpha val
	fflush(stdout);								//clean
	printf("q = %f \n", o.q);					// quality that uphold q < qc
	fflush(stdout);								//clean
	printf("t = %f \n", o.t);					//min time 
	fflush(stdout);								//clean
	for (i = 0; i < o.k; i++) {					//dimantion loop
		printf("w[%d] = %f \n", i, o.w[i]);		// print vector of weights of each procces
	}
}

// print array of output result
void printOutputArr(int numprocs, output_t* oArr) {
	int j;
	for (j = 0; j < numprocs; j++)				// loop of number of procceses
		printOutput(oArr[j]);					// print output for every procces
	fflush(stdout);								// clean
}

// fin minimal time among all the results that were rwturned from the procceses
int findTMin(int numprocs, output_t* oArr) {
	int index = -1, i;
	double tMin = 100.0;						// initialize minimal time by limit
	// parallel by omp
#pragma omp parallel for
	for (i = 0; i < numprocs; i++) {			// loop of number of procceses
		if (oArr[i].t < tMin && oArr[i].q != -1) {	//check if t < limit and q < qc
			tMin = oArr[i].t;						// if time was found write to output
			index = i;	
		}
	}
	return index;									// return number of procces
}

// save outpot to file
void saveToOutputFile(output_t o) {
	int i;
	FILE* f = fopen(OUTPUT_FILE_NAME, "w");			// find file by path
	if (f == NULL)									// check if file exist
	{
		printf("Failed opening the file.\nExiting!\n");	//error message if file was not open
		return;
	}
	if (o.alpha == 1) {								// check alpha
		fputs("Alpha was not found ", f);			// alpha error message
	}
	else {
		fputs("Alpha minimum = ", f);				// print alpha
		fprintf(f, "%lf,", o.alpha);
		fputs("q = ", f);							// print best quality that was reached
		fprintf(f, "%lf \n", o.q);		
		for (i = 0; i < o.k; i++)					// dimantion loop
			fprintf(f, "%lf \n", o.w[i]);;			// print weight vector that belongs to the result
	}

	fclose(f);
}

// free allocation
void freeAll(int* arr, point_t* pointsArr, cudaError_t cudaStatus) {
	free(pointsArr);
	//CUDA:
	free(arr);
	cudaStatus = free_All();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed free cuda!");
	}
}

