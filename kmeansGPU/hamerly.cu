#include "hamerly.h"

std::chrono::duration<double> HamerlyKmeans::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    std::cout << "HamerlyKmeans init" << std::endl;
    auto s = std::chrono::system_clock::now();
    Kmeans::initialize(aX, aK, initialAssignment, aNumThreads);
    auto e = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds1 = e - s;
    std::cout << "kmeans init basic: " << elapsed_seconds1.count() << "\n";

    auto start = std::chrono::system_clock::now();
    gpuErrchk(cudaMalloc(&d_centerCenterDistDiv2, (k * k) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_s, k * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_lower, n * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_upper, n * sizeof(double)));

    gpuErrchk(cudaMemset(d_lower, 0.0, n * sizeof(double)));
    gpuErrchk(cudaMemset(d_centerCenterDistDiv2, 0.0, (k * k) * sizeof(double)));
    initArrDouble << <numBlocksC, blockSizeC >> > (d_upper, INFINITY, nC);
    auto end = std::chrono::system_clock::now();

    return end - start;
}

void HamerlyKmeans::free() {
    std::cout << "hamerly free" << std::endl;

    cudaFree(d_centerCenterDistDiv2);
    cudaFree(d_lower);
    cudaFree(d_s);
    cudaFree(d_upper);
}

void HamerlyKmeans::executeSingleIteration() {
    initArrDouble << <numBlocksC, blockSizeC >> > (d_s, INFINITY, k);
    innerProd << <numBlocksI, blockSizeI >> > (d_centerCenterDistDiv2, d_s, centers->d_data, centers->d, centers->n);
    
#if KFUNCTIONS
    hamerlyFunK << <numBlocksK, blockSizeK >> > (x->d_data, centers->d_data, d_assignment,
        d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, n, d_closest2, d_distances, d_calculated);
    hamCombineK << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment,
        d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, n, d_closest2, d_calculated, d_distances);
#else
    hamerlyFun << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment,
        d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, n, d_closest2, d_countDistances);
#endif

    changeAss << <numBlocksC, blockSizeC >> > (x->d_data, d_assignment, d_closest2, d_clusterSize, sumNewCenters[0]->d_data, d, nC, 0);

    cudaMemcpy(d_converged, &converged, 1 * sizeof(bool), cudaMemcpyHostToDevice);
    elkanMoveCenter << <numBlocksM, blockSizeM >> > (d_centerMovement, d_clusterSize, centers->d_data, sumNewCenters[0]->d_data, d_converged, k, d, nM);
    cudaMemcpy(&converged, d_converged, 1 * sizeof(bool), cudaMemcpyDeviceToHost);

    if (!converged) {
#if SHAREDBOUND
        updateBoundHamShared << <numBlocksSB, 66 >> > (d_lower, d_upper,
            d_centerMovement, d_assignment, k, nSB);
        /*updateBoundHamShared << <numBlocksSB, blockSizeSB >> > (d_lower, d_upper,
            d_centerMovement, d_assignment, k, nSB);*/
#else
        updateBoundHam << <numBlocksB, blockSizeB >> > (d_lower, d_upper, d_centerMovement,
            d_assignment, k, nB);
#endif     
    }
}