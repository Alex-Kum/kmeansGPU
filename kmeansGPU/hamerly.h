#pragma once

#include "kmeans.h"
#include "gpufunctions.cuh"

class HamerlyKmeans : public Kmeans {
public:
    virtual ~HamerlyKmeans() { free(); }
    virtual void free();
    virtual std::chrono::duration<double> initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads);
    virtual std::string getName() const { return "Hamerly"; }

protected:
    void executeSingleIteration();

    double* d_s;
    double* d_centerCenterDistDiv2;
    double* d_lower;
    double* d_upper;

};