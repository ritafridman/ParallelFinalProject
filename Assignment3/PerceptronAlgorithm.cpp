#include "perceptronAlgorithm.h"

// initialize weight to 0
void initWeights(int k, double* weights) {
	int i;
	// use omp
#pragma omp parallel for
	for (i = 0; i < k; i++)					// loop by dimantion (coardinats)
		weights[i] = 0.0;					// initialize weight
}

// classification to A or B
int sign(double num) {						// num that was returned from f function
	if (num >= 0.0)							// check if positive
		return 1;							// A classification
	else
		return -1;							// negative  - > B classification
}

// check sum of weight and point vectors multiplication
double f(double* w, point_t point) {
	int i;
	double sum = 0;										// initialize sum to zero
	for (i = 0; i < point.k - 1; i++) {					// loop by coardinats
		sum = sum + (point.values[i] * w[i]);			// multiple betven weight and point and sum
	}
	sum = sum + w[i];									// multiple last coardint weight and by 1 and sum
	return sum;
}

// update weight vector
void algorithmTraining(int k, point_t point, double* w, double alpha, int sign, double t) {
	int i;
	for (i = 0; i < k - 1; i++) {							// loop by coardinats
		w[i] = w[i] + alpha*(double)sign*point.values[i];	//update weight vector
	}
	w[i] = w[i] + alpha*(double)sign;						// sum last coardinat weight vector
}

output_t calculatePerceptron(int n, point_t* pointsArr, int k, double t, double dt, double tMax, double* alpha, double q, double* qc, int limit, int* arr, int rank, int tRange[])
{
	output_t o;								// initialize output
	int i, j, nMis = 0;						// initialize nMis
	double error;							// check if sign are equal
	t = tRange[0];							// set t as procces start range
	tMax = tRange[1];						// set tMax as procces end range
	q = *qc + 1;							// set for the contiotion to be true
	o.q = q;								// set q of output
	o.k = k;								// set dimantion of output
	o.t = limit;							// set output t as limit
	o.alpha = *alpha;						// set alpha that was taken from data
	cudaError_t cudaStatus;
	double afterSign = 0.0;					// initial after sign value
											
	//Moving every point to its initial position according to the tRange of the procces.
	setInitialPointsPosition(pointsArr, dt, k, n, tRange[0]); 
	//If the q that we found is smaller than qc OR t is bigger than tMax exit loop.
	while ((q > *qc) && (t < tMax)) { 
		initWeights(k, o.w);				// initialize weight to zero
		for (j = 0; j < limit; j++) {		// loop till limit
			nMis = 0;						// nMis count
			for (i = 0; i < n; i++) {								//loop of number of points
				afterSign = sign(f(o.w, pointsArr[i]));				// check sign
				error = pointsArr[i].groupIdentifier - afterSign;	// check if calculation is equal to data
				if (!(error == 0)) {								//go in if need to train
					nMis += 1;										// nMis counter
					// train the weight vector
					algorithmTraining(k, pointsArr[i], o.w, *alpha, pointsArr[i].groupIdentifier, t); 
				}
			}
			// qualiti calculation
			q = (double)nMis / (double)n;

			//chack if true , if yes that is the minimal t , can return output and stop checking.
			if (q < *qc && q < o.q) {			// q smaller than qc and smaller than the smalles we found
				o.q = q;						// set q to output
				o.t = t;						// set t to output
				return o;						// return output
			}
		}
		/* 
		//CUDA: find the nMiss: did not work properly.
		cudaStatus = Calculate(n, o.k, o.w, arr);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "failed!");
		}
		*/

		t += dt;								//advance t by dt

		pointsAdvancement(pointsArr, dt, k, n); //Advance the points by velocity and time
	}

	return o;									// return output				
}

//advenc each point coardinats by velocity and time
void pointsAdvancement(point_t* pointsArray, double dt, int k, int n) {
	//use omp
#pragma omp parallel for
	for (int i = 0; i < n; i++) {											// loop by points
		for (int j = 0; j < k - 1; j++) {									// loop by coardinats
			pointsArray[i].values[j] += pointsArray[i].velocity[j] * dt;	//advance
		}
	}
}

// set each point to its initial position by range
void setInitialPointsPosition(point_t* pointsArray, double dt, int k, int n, int tStart) {
	int steps = tStart / dt;					// calculate steps till time range
	// use omp
#pragma omp parallel for
	for (int i = 0; i < n; i++) {				// loop of points
		for (int j = 0; j < k - 1; j++) {		// loop of dimention
			for (int k = 0; k < steps; k++)		// loop of steps till time range
				pointsArray[i].values[j] += pointsArray[i].velocity[j] * dt;	//advance each coardinat point
		}
	}
}
