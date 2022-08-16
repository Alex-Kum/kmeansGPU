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

class LloydKmeans : public Kmeans {
public:
    //virtual ~LloydKmeans() { free(); }
    virtual std::string getName() const { return "lloyd"; }

protected:
    void executeSingleIteration();
};