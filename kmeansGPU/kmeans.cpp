/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * https://github.com/kuanhsunchen/ElkanOpt
 *
 * I added GPU Code
 */

#include "kmeans.h"
#include <cassert>
#include <cmath>

#define VERIFY_ASSIGNMENTS 0

Kmeans::Kmeans() : x(NULL), n(0), k(0), d(0), converged(false), counter(0),
clusterSize(NULL), assignment(NULL) {
    //std::cout << "kmeans konstructor" << std::endl;
#ifdef COUNT_DISTANCES
    numDistances = 0;
#endif
}

void Kmeans::free() {
    std::cout << "free" << std::endl;
    cudaFree(d_centerMovement);
    for (int t = 0; t < 1; ++t) {
        if (clusterSize[t] != nullptr)
            delete clusterSize[t];
    }
    cudaFree(d_clusterSize);
    cudaFree(d_assignment);
    cudaFree(d_closest2);
    cudaFree(d_converged);
    cudaFree(d_countDistances);
#if KFUNCTIONS
    cudaFree(d_calculated);
    cudaFree(d_distances);
#endif
    delete[] clusterSize;
    delete[] centerMovement;

    clusterSize = NULL;
    centerMovement = NULL;
    assignment = NULL;
    n = k = d = 0;
}

std::chrono::duration<double> Kmeans::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    std::cout << "kmeans init" << std::endl;
    converged = false;
    x = aX;
    n = x->n;
    d = x->d;
    k = aK;

    nC = n;
    blockSizeC = 3 * 32;
    numBlocksC = (nC + blockSizeC - 1) / blockSizeC;

    nK = n * k;
    blockSizeK = 3 * 32;
    numBlocksK = (nK + blockSizeK - 1) / blockSizeK;

    nM = k;
    blockSizeM = 1 * 32;
    numBlocksM = (nM + blockSizeM - 1) / blockSizeM;

    nI = k * k;
    blockSizeI = 1 * 32;
    numBlocksI = (nI + blockSizeI - 1) / blockSizeI;

    nB = n;
    blockSizeB = 1 * 32;
    numBlocksB = (nB + blockSizeB - 1) / blockSizeB;

    nSB = n;
    blockSizeSB = 66;
    numBlocksSB = (nSB + blockSizeSB - 1) / blockSizeSB;

    assignment = initialAssignment;
    gpuErrchk(cudaMalloc(&d_assignment, n * sizeof(unsigned short)));
    gpuErrchk(cudaMalloc(&d_closest2, n * sizeof(unsigned short)));
    gpuErrchk(cudaMalloc(&d_converged, 1 * sizeof(bool)));

    gpuErrchk(cudaMalloc(&d_countDistances, 1 * sizeof(unsigned long long int)));
    gpuErrchk(cudaMemset(d_countDistances, 0, 1 * sizeof(unsigned long long int)));

#if KFUNCTIONS
    gpuErrchk(cudaMalloc(&d_calculated, n * k * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_distances, (n * k) * sizeof(double)));
#endif

    centerMovement = new double[k];
    gpuErrchk(cudaMalloc(&d_centerMovement, k * sizeof(double)));
    gpuErrchk(cudaMemset(d_centerMovement, 0, k * sizeof(double)));

    clusterSize = new int* [1];
    gpuErrchk(cudaMalloc(&d_clusterSize, k * sizeof(int)));

    for (int t = 0; t < 1; ++t) {
        clusterSize[t] = new int[k];

        std::fill(clusterSize[t], clusterSize[t] + k, 0);
        for (int i = 0; i < n; ++i) {
            assert(assignment[i] < k);
            ++clusterSize[t][assignment[i]];
        }
    }

    centers = new Dataset(k, d);
    sumNewCenters = new Dataset * [1];
    centers->fill(0.0);

    for (int t = 0; t < 1; ++t) {
        sumNewCenters[t] = new Dataset(k, d, false);
        sumNewCenters[t]->fill(0.0);

        for (int i = 0; i < n; ++i) {
            addVectors(sumNewCenters[t]->data + assignment[i] * d, x->data + i * d, d); //initialize
        }
    }

    move_centers();


    gpuErrchk(cudaMemcpy(x->d_data, x->data, (n * d) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(centers->d_data, centers->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(sumNewCenters[0]->d_data, sumNewCenters[0]->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_clusterSize, clusterSize[0], k * sizeof(int), cudaMemcpyHostToDevice));

    return std::chrono::system_clock::now()- std::chrono::system_clock::now();
}

void Kmeans::changeAssignment(int xIndex, int closestCluster, int threadId) {
    --clusterSize[threadId][assignment[xIndex]];
    ++clusterSize[threadId][closestCluster];

    assignment[xIndex] = closestCluster;
}

int Kmeans::move_centers() {
    int furthestMovingCenter = 0;
    for (int j = 0; j < k; ++j) {
        centerMovement[j] = 0.0;
        int totalClusterSize = 0;
        for (int t = 0; t < 1; ++t) {
            totalClusterSize += clusterSize[t][j];
        }
        if (totalClusterSize > 0) {
            for (int dim = 0; dim < d; ++dim) {
                double z = 0.0;
                for (int t = 0; t < 1; ++t) {
                    z += (*sumNewCenters[t])(j, dim);
                }
                z /= totalClusterSize;
                centerMovement[j] += (z - (*centers)(j, dim)) * (z - (*centers)(j, dim));//calculate distance
                (*centers)(j, dim) = z; //update new centers
            }
        }
        centerMovement[j] = sqrt(centerMovement[j]);

        if (centerMovement[furthestMovingCenter] < centerMovement[j]) {
            furthestMovingCenter = j;
        }
    }

    return furthestMovingCenter;
}

int Kmeans::run(int maxIterations) {
    int iterations = 0;
    converged = false;
    cudaMemcpy(d_converged, &converged, 1 * sizeof(bool), cudaMemcpyHostToDevice);
    while (iterations < maxIterations && !converged) {
        converged = true;
        executeSingleIteration();
        iterations++;
       // std::cout << "iteration: " << iterations << std::endl;
    }
    //std::cout << "da" << std::endl;
    unsigned long long int cDist;
    cudaMemcpy(&cDist, d_countDistances, 1 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    std::cout << "distance calculations: " << cDist << std::endl;
   // std::cout << "wuff" << std::endl;
    return iterations;    
}


void Kmeans::verifyAssignment(int iteration, int startNdx, int endNdx) const {
#ifdef VERIFY_ASSIGNMENTS
    //for (int i = startNdx; i < endNdx; ++i) {
    //    // keep track of the squared distance and identity of the closest-seen
    //    // cluster (so far)
    //    int closest = assignment[i];
    //    double closest_dist2 = pointCenterDist2(i, closest);
    //    double original_closest_dist2 = closest_dist2;
    //    // look at all centers
    //    for (int j = 0; j < k; ++j) {
    //        if (j == closest) {
    //            continue;
    //        }
    //        double d2 = pointCenterDist2(i, j);

    //        // determine if we found a closer center
    //        if (d2 < closest_dist2) {
    //            closest = j;
    //            closest_dist2 = d2;
    //        }
    //    }

    //    // if we have found a discrepancy, then print out information and crash
    //    // the program
    //    if (closest != assignment[i]) {
    //        std::cerr << "assignment error:" << std::endl;
    //        std::cerr << "iteration             = " << iteration << std::endl;
    //        std::cerr << "point index           = " << i << std::endl;
    //        std::cerr << "closest center        = " << closest << std::endl;
    //        std::cerr << "closest center dist2  = " << closest_dist2 << std::endl;
    //        std::cerr << "assigned center       = " << assignment[i] << std::endl;
    //        std::cerr << "assigned center dist2 = " << original_closest_dist2 << std::endl;
    //        assert(false);
    //    }
    //}
#endif
}