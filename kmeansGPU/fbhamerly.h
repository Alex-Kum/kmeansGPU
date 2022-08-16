#pragma once

#include "kmeans.h"
#include "gpufunctions.cuh"

class FBHamerly : public Kmeans {
public:
    virtual ~FBHamerly() { free(); }
    virtual void free();
    virtual std::chrono::duration<double> initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads);
    virtual std::string getName() const { return "fbhamerly"; }

protected:
    void executeSingleIteration();

    double* d_oldcenter2newcenterDis;
    double* d_maxoldcenter2newcenterDis;
    double* d_oldcenters;
    //bool* d_calculated;
    double* d_ub_old;

    double* d_s;
    double* d_centerCenterDistDiv2;
    double* d_lower;
    double* d_upper;
};