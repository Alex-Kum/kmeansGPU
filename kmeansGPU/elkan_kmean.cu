#include "elkan_kmean.h"

std::chrono::duration<double> ElkanKmeans::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    std::cout << "ElkanKmeans init" << std::endl;
    Kmeans::initialize(aX, aK, initialAssignment, aNumThreads);

    auto start = std::chrono::system_clock::now();
    gpuErrchk(cudaMalloc(&d_centerCenterDistDiv2, (k * k) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_s, k * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_upper, n * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_lower, (n * k) * sizeof(double)));

    gpuErrchk(cudaMemset(d_lower, 0.0, (n * k) * sizeof(double)));
    gpuErrchk(cudaMemset(d_centerCenterDistDiv2, 0.0, (k * k) * sizeof(double)));
    gpuErrchk(cudaMemset(d_centerCenterDistDiv2, 0.0, (k * k) * sizeof(double)));
    initArrDouble<< <numBlocksC, blockSizeC >> > (d_upper, INFINITY, nC);

    auto end = std::chrono::system_clock::now();

    return end - start;
}

void ElkanKmeans::free() {
    std::cout << "elkan free" << std::endl;

    cudaFree(d_centerCenterDistDiv2);
    cudaFree(d_lower);
    cudaFree(d_s);
    cudaFree(d_upper);
}

void ElkanKmeans::executeSingleIteration() {
    //initArrDouble << <numBlocksC, blockSizeC >> > (d_s, INFINITY, k);
    cudaMemset(d_s, std::numeric_limits<double>::max(), k * sizeof(double));
    innerProd << <numBlocksI, blockSizeI >> > (d_centerCenterDistDiv2, d_s, centers->d_data, centers->d, centers->n);
#if KFUNCTIONS
    elkanFunK << <numBlocksK, blockSizeK >> > (x->d_data, centers->d_data, d_assignment,
        d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, n, d_closest2, d_calculated, d_distances);

    elkCombineK << <numBlocksC, blockSizeC >> > (d_lower, d_upper, k, nC, d_closest2, d_calculated, d_distances);
#else
    elkanFun << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment,
        d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, n, d_closest2, 0, d_countDistances);
#endif
    changeAss << <numBlocksC, blockSizeC >> > (x->d_data, d_assignment, d_closest2, d_clusterSize, sumNewCenters[0]->d_data, d, nC, 0);

    cudaMemcpy(d_converged, &converged, 1 * sizeof(bool), cudaMemcpyHostToDevice);
    elkanMoveCenter << <numBlocksM, blockSizeM >> > (d_centerMovement, d_clusterSize, centers->d_data, sumNewCenters[0]->d_data, d_converged, k, d, nM);
    cudaMemcpy(&converged, d_converged, 1 * sizeof(bool), cudaMemcpyDeviceToHost);

    if (!converged) {
#if SHAREDBOUND
        updateBoundShared << <numBlocksSB, 66 >> > (d_lower, d_upper,
            d_centerMovement, d_assignment, k, nSB);
        /*updateBoundShared << <numBlocksSB, blockSizeSB >> > (d_lower, d_upper, 
            d_centerMovement, d_assignment, k, nSB);*/
        /*updateBound << <numBlocksB, blockSizeB >> > (d_lower, d_upper,
            d_centerMovement, d_assignment, k, nB);*/
#else
        updateBound << <numBlocksB, blockSizeB >> > (d_lower, d_upper,
            d_centerMovement, d_assignment, k, nB);
#endif
    }
}