#pragma once

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * https://github.com/kuanhsunchen/ElkanOpt
 *
 * I added GPU Code
 *
 * Elkan's k-means algorithm that uses k lower bounds per point to prune
 * distance calculations.
 */


#include "kmeans.h"
#include "gpufunctions.cuh"

class ElkanKmeans : public Kmeans {
public:
    virtual ~ElkanKmeans() { free(); }
    virtual std::string getName() const { return "elkan"; }
    virtual void free();
    virtual std::chrono::duration<double> initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads);

protected:
    void executeSingleIteration();

    double* d_centerCenterDistDiv2;
    double* d_s;
    double* d_lower;    
    double* d_upper;
};