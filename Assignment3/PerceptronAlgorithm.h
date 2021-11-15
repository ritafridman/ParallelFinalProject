#pragma once

#include "basicFunctions.h"

//initialize weight to 0
void initWeights(int k, double* weights);

// find sum of the weight and point value multiplication
double f(double* w, point_t point);

// classification to A or B
int sign(double num);

// updete weight vector
void algorithmTraining(int k, point_t point, double* w, double alpha, int error, double t);

// set initial point position to every procces
void setInitialPointsPosition(point_t* pointsArray, double dt, int k, int n, int tStart);

// advance every point by time and velocity
void pointsAdvancement(point_t* pointsArray, double dt, int k, int n);

// calculate perceptron algorithm
output_t calculatePerceptron(int n, point_t* pointsArray, int k, double t, double dt, double tMax, double* alpha, double q, double* qc, int limit, int* arr, int rank, int tRange[]);