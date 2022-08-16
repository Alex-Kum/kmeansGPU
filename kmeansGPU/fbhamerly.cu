#include "fbhamerly.h"

std::chrono::duration<double> FBHamerly::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    std::cout << "FBHamerly init" << std::endl;
    Kmeans::initialize(aX, aK, initialAssignment, aNumThreads);

    auto start = std::chrono::system_clock::now();
    gpuErrchk(cudaMalloc(&d_oldcenter2newcenterDis, (k * k) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_maxoldcenter2newcenterDis, k * sizeof(double)));      
    gpuErrchk(cudaMalloc(&d_ub_old, n * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_oldcenters, (k * d) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_centerCenterDistDiv2, (k * k) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_s, k * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_upper, n * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_lower, n * sizeof(double)));

    gpuErrchk(cudaMemset(d_lower, 0.0, n * sizeof(double)));
    gpuErrchk(cudaMemset(d_centerCenterDistDiv2, 0.0, (k * k) * sizeof(double)));
    gpuErrchk(cudaMemset(d_oldcenter2newcenterDis, 0.0, (k * k) * sizeof(double)));
    gpuErrchk(cudaMemset(d_ub_old, 0.0, n * sizeof(double)));
    initArrDouble << <numBlocksC, blockSizeC >> > (d_upper, INFINITY, nC);
    auto end = std::chrono::system_clock::now();

    return end - start;
}

void FBHamerly::free() {
    std::cout << "fbhamerly free" << std::endl;

    cudaFree(d_centerCenterDistDiv2);
    cudaFree(d_lower);
    cudaFree(d_s);
    cudaFree(d_upper);
    cudaFree(d_oldcenter2newcenterDis);
    cudaFree(d_ub_old);
    cudaFree(d_oldcenters);
    cudaFree(d_maxoldcenter2newcenterDis);
}

void FBHamerly::executeSingleIteration() {
    initArrDouble << <numBlocksC, blockSizeC >> > (d_s, INFINITY, k);
    innerProd << <numBlocksI, blockSizeI >> > (d_centerCenterDistDiv2, d_s, centers->d_data, centers->d, centers->n);

#if KFUNCTIONS
    fbhamerlyFunK << <numBlocksK, blockSizeK >> > (x->d_data, centers->d_data, d_assignment,
        d_lower, d_upper, d_s, d_centerCenterDistDiv2, d_maxoldcenter2newcenterDis, d_ub_old, k, d, nC, d_closest2, d_distances, d_calculated);
    hamCombineK << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment,
        d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, n, d_closest2, d_calculated, d_distances);

#else
    elkanFunFBHam << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment,
        d_lower, d_upper, d_s, d_centerCenterDistDiv2, d_maxoldcenter2newcenterDis, d_ub_old, k, d, nC, d_closest2, d_countDistances);
#endif

    
    changeAss << <numBlocksC, blockSizeC >> > (x->d_data, d_assignment, d_closest2, d_clusterSize, sumNewCenters[0]->d_data, d, nC, 0);
    
    cudaMemcpy(d_converged, &converged, 1 * sizeof(bool), cudaMemcpyHostToDevice);
    elkanMoveCenterFB << <numBlocksM, blockSizeM >> > (d_centerMovement, d_clusterSize,
        centers->d_data, sumNewCenters[0]->d_data, d_oldcenters, d_converged, k, d, nM);
    cudaMemcpy(&converged, d_converged, 1 * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaMemset(d_oldcenter2newcenterDis, 0.0, (k * k) * sizeof(double));
    elkanFBMoveAddition << <numBlocksI, blockSizeI >> > (d_oldcenters, d_oldcenter2newcenterDis, centers->d_data, d, k, centers->n);
    
    elkanFBMoveAdditionHam << <centers->n, 1 >> > (d_oldcenters, d_oldcenter2newcenterDis, d_maxoldcenter2newcenterDis, k, centers->n);

    if (!converged) {
#if SHAREDBOUND
        updateBoundHamFBShared << <numBlocksSB, 258>> > (d_lower, d_upper, d_ub_old,
            d_centerMovement, d_assignment, k, nSB);
        /*updateBoundHamFBShared << <numBlocksSB, blockSizeSB >> > (d_lower, d_upper, d_ub_old,
            d_centerMovement, d_assignment, k, nSB);*/
#else
        updateBoundFBHam << <numBlocksB, blockSizeB >> > (d_lower, d_upper, d_ub_old,
            d_centerMovement, d_assignment, k, nB);
#endif    
        
    }
}