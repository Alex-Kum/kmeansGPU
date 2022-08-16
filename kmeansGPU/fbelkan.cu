#include "fbelkan.h"

std::chrono::duration<double> FBElkanKmeans::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    std::cout << "FBElkanKmeans init" << std::endl;
    Kmeans::initialize(aX, aK, initialAssignment, aNumThreads);

    auto start = std::chrono::system_clock::now();
    gpuErrchk(cudaMalloc(&d_centerCenterDistDiv2, (k * k) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_oldcenter2newcenterDis, (k * k) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_s, k * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_upper, n * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_lower, (n * k) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_ub_old, n * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_oldcenters, (k * d) * sizeof(double)));

    gpuErrchk(cudaMemset(d_lower, 0.0, (n * k) * sizeof(double)));
    gpuErrchk(cudaMemset(d_centerCenterDistDiv2, 0.0, (k * k) * sizeof(double)));
    gpuErrchk(cudaMemset(d_oldcenter2newcenterDis, 0.0, (k * k) * sizeof(double)));
    gpuErrchk(cudaMemset(d_ub_old, 0.0, n * sizeof(double)));
    initArrDouble << <numBlocksC, blockSizeC >> > (d_upper, INFINITY, nC);
    auto end = std::chrono::system_clock::now();

    return end - start;
}

void FBElkanKmeans::free() {
    std::cout << "fbelkan free" << std::endl;

    cudaFree(d_centerCenterDistDiv2);
    cudaFree(d_lower);
    cudaFree(d_s);
    cudaFree(d_upper);
    cudaFree(d_oldcenter2newcenterDis);
    cudaFree(d_ub_old);
    cudaFree(d_oldcenters);
}

void FBElkanKmeans::executeSingleIteration() {
    initArrDouble << <numBlocksC, blockSizeC >> > (d_s, INFINITY, k);
    innerProd << <numBlocksI, blockSizeI >> > (d_centerCenterDistDiv2, d_s, centers->d_data, centers->d, centers->n);

#if KFUNCTIONS
    fbelkanFunK << <numBlocksK, blockSizeK >> > (x->d_data, centers->d_data, d_assignment,
        d_lower, d_upper, d_s, d_oldcenter2newcenterDis, d_ub_old, d_calculated, n, d, k, d_closest2, d_centerCenterDistDiv2);

    elkCombineK << <numBlocksC, blockSizeC >> > (d_lower, d_upper, k, nC, d_closest2, d_calculated, d_distances);
#else
    elkanFunFB << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment,
        d_lower, d_upper, d_s, d_centerCenterDistDiv2, d_oldcenter2newcenterDis, d_ub_old, k, d, nC, d_closest2, d_countDistances);
#endif
    /*elkanFun << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment,
        d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, n, d_closest2, 0, d_countDistances);*/
    changeAss << <numBlocksC, blockSizeC >> > (x->d_data, d_assignment, d_closest2, d_clusterSize, sumNewCenters[0]->d_data, d, nC, 0);

    cudaMemcpy(d_converged, &converged, 1 * sizeof(bool), cudaMemcpyHostToDevice);
    elkanMoveCenterFB << <numBlocksM, blockSizeM >> > (d_centerMovement, d_clusterSize, 
        centers->d_data, sumNewCenters[0]->d_data, d_oldcenters, d_converged, k, d, nM);
    cudaMemcpy(&converged, d_converged, 1 * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaMemset(d_oldcenter2newcenterDis, 0.0, (k * k) * sizeof(double));
    elkanFBMoveAddition << <numBlocksI, blockSizeI >> > (d_oldcenters, d_oldcenter2newcenterDis, centers->d_data, d, k, centers->n);

    if (!converged) {
#if SHAREDBOUND
        updateBoundFBShared << <numBlocksSB, 66 >> > (d_lower, d_upper, d_ub_old,
            d_centerMovement, d_assignment, k, nSB);
        /*updateBoundFBShared << <numBlocksSB, blockSizeSB >> > (d_lower, d_upper, d_ub_old,
            d_centerMovement, d_assignment, k, nSB);*/
#else
        updateBoundFB << <numBlocksB, blockSizeB >> > (d_lower, d_upper, d_ub_old,
            d_centerMovement, d_assignment, k, nB);
#endif
       
    }
}