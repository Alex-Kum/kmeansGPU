#include "moelkan.h"

std::chrono::duration<double> MOElkanKmeans::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    std::cout << "MOElkanKmeans init" << std::endl;
    auto s = std::chrono::system_clock::now();
    Kmeans::initialize(aX, aK, initialAssignment, aNumThreads);
    auto e = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds1 = e - s;
    std::cout << "kmeans init basic: " << elapsed_seconds1.count() << "\n";

    auto start = std::chrono::system_clock::now();
    gpuErrchk(cudaMalloc(&d_centerCenterDistDiv2, (k * k) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_oldcenter2newcenterDis, (k * k) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_oldcenterCenterDistDiv2, (k * k) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_s, k * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_upper, n * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_ub_old, n * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_oldcenters, (k * d) * sizeof(double)));

    gpuErrchk(cudaMemset(d_centerCenterDistDiv2, 0.0, (k * k) * sizeof(double)));
    gpuErrchk(cudaMemset(d_oldcenter2newcenterDis, 0.0, (k * k) * sizeof(double)));
    gpuErrchk(cudaMemset(d_oldcenterCenterDistDiv2, 0.0, (k * k) * sizeof(double)));
    initArrDouble << <numBlocksC, blockSizeC >> > (d_ub_old, INFINITY, nC);
    initArrDouble << <numBlocksC, blockSizeC >> > (d_upper, INFINITY, nC);
    auto end = std::chrono::system_clock::now();

    return end - start;
}

void MOElkanKmeans::free() {
    std::cout << "moelkan free" << std::endl;

    cudaFree(d_centerCenterDistDiv2);
    cudaFree(d_oldcenterCenterDistDiv2);
    cudaFree(d_s);
    cudaFree(d_upper);
    cudaFree(d_oldcenter2newcenterDis);
    cudaFree(d_ub_old);
    cudaFree(d_oldcenters);
}

void MOElkanKmeans::executeSingleIteration() {
    initArrDouble << <numBlocksC, blockSizeC >> > (d_s, INFINITY, k);
    innerProdMO << <numBlocksI, blockSizeI >> > (d_centerCenterDistDiv2, d_oldcenterCenterDistDiv2, d_s, centers->d_data, centers->d, k, centers->n);
    
#if KFUNCTIONS
    moelkanFunK << <numBlocksK, blockSizeK >> > (x->d_data, centers->d_data, d_assignment, d_distances, d_upper, d_s, d_oldcenter2newcenterDis, d_ub_old,
        d_calculated, n, d, k, d_closest2, d_centerCenterDistDiv2, d_oldcenterCenterDistDiv2, d_centerMovement);

    elkCombineK << <numBlocksC, blockSizeC >> > (d_distances, d_upper, k, nC, d_closest2, d_calculated, d_distances);
#else
    elkanFunMO << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment, d_upper, d_s, d_centerCenterDistDiv2,
        d_oldcenter2newcenterDis, d_oldcenterCenterDistDiv2, d_ub_old, d_centerMovement, k, d, nC, d_closest2, d_countDistances);
#endif

    
    changeAss << <numBlocksC, blockSizeC >> > (x->d_data, d_assignment, d_closest2, d_clusterSize, sumNewCenters[0]->d_data, d, nC, 0);

    cudaMemcpy(d_converged, &converged, 1 * sizeof(bool), cudaMemcpyHostToDevice);
    elkanMoveCenterFB << <numBlocksM, blockSizeM >> > (d_centerMovement, d_clusterSize,
        centers->d_data, sumNewCenters[0]->d_data, d_oldcenters, d_converged, k, d, nM);
    cudaMemcpy(&converged, d_converged, 1 * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaMemset(d_oldcenter2newcenterDis, 0.0, (k * k) * sizeof(double));
    elkanFBMoveAddition << <numBlocksI, blockSizeI >> > (d_oldcenters, d_oldcenter2newcenterDis, centers->d_data, d, k, centers->n);

    if (!converged) {
        updateBoundMO << <numBlocksB, blockSizeB >> > (d_upper, d_ub_old, d_centerMovement, d_assignment, nB);
    }
}