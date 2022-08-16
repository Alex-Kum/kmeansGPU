#pragma once

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * Kmeans is an abstract base class for algorithms which implement Lloyd's
 * k-means algorithm. Subclasses provide functionality in the "runThread()"
 * method.
 *
 * https://github.com/kuanhsunchen/ElkanOpt
 *
 * I added GPU Code
 */

#include "dataset.h"
#include <limits>
#include <string>
#include "general_functions.h"
#include <chrono>

#define KFUNCTIONS 1
#define SHAREDBOUND 0

class Kmeans {
public:
    // Construct a K-means object to operate on the given dataset
    Kmeans();
    virtual ~Kmeans() { free(); }

    // This method kicks off the threads that do the clustering and run
    // until convergence (or until reaching maxIterations). It returns the
    // number of iterations performed.
    int run(int aMaxIterations = std::numeric_limits<int>::max());   

    // Get the cluster assignment for the given point index.
    int getAssignment(int xIndex) const { return assignment[xIndex]; }
    int move_centers();

    virtual void executeSingleIteration() {};

    // Initialize the algorithm at the beginning of the run(), with the
    // given data and initial assignment. The parameter initialAssignment
    // will be modified by this algorithm and will at the end contain the
    // final assignment of clusters.
    virtual std::chrono::duration<double> initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads);

    // Free all memory being used by the object.
    virtual void free();

    // This method verifies that the current assignment is correct, by
    // checking every point-center distance. For debugging.
    virtual void verifyAssignment(int iteration, int startNdx, int endNdx) const;

    // Get the name of this clustering algorithm (to be overridden by
    // subclasses).
    virtual std::string getName() const = 0; 
    virtual Dataset const* getCenters() const { return NULL; }

protected:
    // The dataset to cluster.
    const Dataset* x;

    Dataset* centers;
    Dataset** sumNewCenters;

    // Local copies for convenience.
    int n, k, d;
    int counter;

    // To communicate (to all threads) that we have converged.
    bool converged;
    bool* d_converged;

    unsigned long long int* d_countDistances;
    

    // Keep track of how many points are in each cluster, divided over each
    // thread.
    int** clusterSize;
    int* d_clusterSize;

    // centerMovement is computed in move_centers() and used to detect
    // convergence (if max(centerMovement) == 0.0) and update point-center
    // distance bounds (in subclasses that use them).
    double* centerMovement;
    double* d_centerMovement;

    // For each point in x, keep which cluster it is assigned to. By using a
    // short, we assume a limited number of clusters (fewer than 2^16).
    unsigned short* assignment;
    unsigned short* d_assignment;
    unsigned short* d_closest2;

#if KFUNCTIONS
    bool* d_calculated;
    double* d_distances;
#endif

    int nC, blockSizeC, numBlocksC;
    int nM, blockSizeM, numBlocksM;
    int nI, blockSizeI, numBlocksI;
    int nB, blockSizeB, numBlocksB;
    int nSB, blockSizeSB, numBlocksSB;
    int nK, blockSizeK, numBlocksK;

    // Assign point at xIndex to cluster newCluster, working within thread threadId.
    virtual void changeAssignment(int xIndex, int newCluster, int threadId);
};