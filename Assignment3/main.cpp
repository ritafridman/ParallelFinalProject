#include "basicFunctions.h"
#include "PerceptronAlgorithm.h"

int main(int argc, char *argv[]) {
	//CUDA:
	int* arr;
	cudaError_t cudaStatus;

	int rank, numprocs, n, k, limit, i;					// pharm initialize
	int tRange[2] = { 0,0 };							// time range 
	point_t* pointsArr;									// array of points
	output_t* outputArr;								// output array
	output_t output;									// output
	double dt, tMax, alpha, qc, t = 0, t0 = 0, t1 = 0, q = 0;		// pharm initialize

	MPI_Datatype PointMPIType;							// point Data Type
	MPI_Datatype OutputMPIType;							// output Data Type
	MPI_Status status;									//MPI status
	MPI_Init(&argc, &argv);								// initialize MPI
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);			//commit size
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);				//comit rank
	createPointDataType(&PointMPIType);					// create point Data Type
	createOutputDataType(&OutputMPIType);				// create output Data Type

	if (rank == MASTER)									// check if procces 0
	{
		t0 = MPI_Wtime();								// start MPI
		// get input from data file
		pointsArr = getInputFromFile(INPUT_FILE_NAME, &n, &k, &dt, &tMax, &alpha, &limit, &qc);
		
		if (numprocs > (tMax / dt) + 1) {				// check if num of procs reached its limit
			printf("Please run with max of %d processes.\n", (int)(tMax / dt) + 1); //error message
			MPI_Abort(MPI_COMM_WORLD, 1);				//abort work
		}

		if (pointsArr == NULL) {							//check if null
			printf("Failed to read input from file.\n");	//error message
			MPI_Abort(MPI_COMM_WORLD, 1);					//abort work
			fflush(stdout);									//clean
		}

		//each point return its result to this array 
		outputArr = (output_t*)calloc(numprocs, sizeof(output_t));
	}

	//send k , limit , alpha , qc , dt and tMax
	MPI_Bcast(&k, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&limit, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&alpha, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&qc, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&dt, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&tMax, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

	//Master send the size of points to proccess
	if (rank == MASTER) {					// check if master
		//ise omp
#pragma omp parallel for
		for (i = 1; i < numprocs; i++) {
			MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
	}

	//Each process gets the size of the array and calloc it
	else {
		MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);		//get array size
		pointsArr = (point_t*)calloc(n, sizeof(point_t));				//allocate 
	}

	arr = (int*)calloc(n, sizeof(int));

	//master send point array and time range to each procces
	if (rank == MASTER) {
		sentPointsAndRangeByT(n, numprocs, pointsArr, tRange, PointMPIType, dt, tMax);
	}

	else {
		// get point array and time range from the master
		MPI_Recv(pointsArr, n, PointMPIType, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(tRange, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&tRange[1], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
	}

	//CUDA: Each process keeps its own chunk of data in cuda:
	//cudaStatus = Save_Array_Points_And_Weights(pointsArr, n, output.w, k);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "failed!");
	//}
	// sccatter output , calculate algorithm and gather
	MPI_Scatter(outputArr, 1, OutputMPIType, &output, 1, OutputMPIType, MASTER, MPI_COMM_WORLD);
	output = calculatePerceptron(n, pointsArr, k, t, dt, tMax, &alpha, q, &qc, limit, arr, rank, tRange);
	MPI_Gather(&output, 1, OutputMPIType, outputArr, 1, OutputMPIType, MASTER, MPI_COMM_WORLD);
	//free alocation
	freeAll(arr, pointsArr, cudaStatus);

	//if master then write to output file
	if (rank == MASTER) {
		int index = findTMin(numprocs, outputArr);
		saveToOutputFile(outputArr[index]);
		t1 = MPI_Wtime();
		printf("Elapsed time: %f\n", t1 - t0);
		fflush(stdout);
		if (index == -1)
			printf("Time was not found\n");
		else
			printOutput(outputArr[index]);
		free(outputArr);
	}

	MPI_Finalize();
}

//get input from file
point_t* getInputFromFile(char* fileName, int* n, int* k, double* dt, double* tMax, double* alpha, int* limit, double* qc)
{
	point_t* pointsArr;
	point_t currentPoint;
	FILE* f = fopen(fileName, "r");
	*n = 0;
	int i, j, l;

	if (f == NULL) {										// check if open file succeeded.
		printf("File failed to open. Exiting...\n");
		return NULL;
	}

	else { 
		fscanf(f, "%d", n);

		if (*n > MAX_SIZE_POINTS || *n < MIN_SIZE_POINTS) {
			printf("Number of points should be between %d and %d .\n", MIN_SIZE_POINTS, MAX_SIZE_POINTS);
			return NULL;
		}
		
		fscanf(f, "%d", k);
		if (*k > MAX_DIM) {
			printf("Dimension shouldn't be higher than %d .\n", MAX_DIM);
			return NULL;
		}
		*k = *k + 1;		//bias
		currentPoint.k = *k;
		fscanf(f, "%lf", dt);
		fscanf(f, "%lf", tMax);
		fscanf(f, "%lf", alpha);

		//check the limit
		fscanf(f, "%d", limit);
		if (*limit > MAX_LIMIT) {
			printf("Limit shouldn't	be higher than %d .\n", MAX_LIMIT);
			return NULL;
		}
		fscanf(f, "%lf", qc);
		// allocating the points array
		pointsArr = (point_t*)calloc(*n, sizeof(point_t));

		fflush(stdout);
		if (pointsArr == NULL) {
			printf("Failed allocating memory! \n");
			return NULL;
		}

		//read all points deatils
		for (i = 0; i < *n; i++) {
			// allocating the inputs array
			for (j = 0; j < (*k) - 1; j++) {
				fscanf(f, "%lf", &currentPoint.values[j]);
			}

			currentPoint.values[j] = 1;

			for (l = 0; l < (*k) - 1; l++) {
				fscanf(f, "%lf", &currentPoint.velocity[l]);
			}
			
			fscanf(f, "%d", &currentPoint.groupIdentifier);
			pointsArr[i] = currentPoint;
		}
		fclose(f);
	}
	return pointsArr;
}

//create point data type
void createPointDataType(MPI_Datatype* PointMPIType) {
	MPI_Datatype types[4] = { MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT };
	int blocklengths[4] = { 20, 20, 1, 1 };
	const int nitems = 4;
	MPI_Aint offsets[4];

	offsets[0] = offsetof(point_t, values);
	offsets[1] = offsetof(point_t, velocity);
	offsets[2] = offsetof(point_t, groupIdentifier);
	offsets[3] = offsetof(point_t, k);

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, PointMPIType);
	MPI_Type_commit(PointMPIType);
}

//create output data type
void createOutputDataType(MPI_Datatype* OutputMPIType) {
	MPI_Datatype type[5] = { MPI_DOUBLE ,MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
	int blocklen[5] = { 20, 1, 1, 1, 1 };
	MPI_Aint offsets[5];
	output_t o;
	
	//use opm
#pragma omp parallel
	{
		offsets[0] = offsetof(output_t, w);
		offsets[1] = offsetof(output_t, alpha);
		offsets[2] = offsetof(output_t, q);
		offsets[3] = offsetof(output_t, t);
		offsets[4] = offsetof(output_t, k);
	}

	MPI_Type_create_struct(5, blocklen, offsets, type, OutputMPIType);
	MPI_Type_commit(OutputMPIType);
}

// master sends points and range
void sentPointsAndRangeByT(int n, int numprocs, point_t* pointsArr, int* tRange, MPI_Datatype PointMPIType, double dt, double tMax) {
	int i = 0, j = 0, k = 0, tStart = 0, tEnd = 0, helper = tMax / dt, max = tMax;
	for (i = 1; i < numprocs; i++) {
		MPI_Send(pointsArr, n, PointMPIType, i, 0, MPI_COMM_WORLD);
	}

	tStart = 0;
	tRange[0] = tStart;
	tEnd = tStart + ((helper / numprocs))*helper;
	tRange[1] = tEnd;
	max -= (tEnd - tStart);
	for (j = 1; j < numprocs; j++) {
		tStart = tEnd;
		MPI_Send(&tStart, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
		tEnd = (tStart + ((helper / numprocs)*helper));
		max -= (tEnd - tStart);
		if (j == numprocs - 1)
			if (max != 0)
				tEnd += max;
		MPI_Send(&tEnd, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
	}
}


