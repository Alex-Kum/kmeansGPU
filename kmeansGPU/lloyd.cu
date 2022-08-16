#include "lloyd.h"

void LloydKmeans::executeSingleIteration() {
#if KFUNCTIONS
    lloydFunK << <numBlocksK, blockSizeK >> > (x->d_data, centers->d_data, k, d, n, d_closest2, d_distances);
    lloydCombineK << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, k, n, d_closest2, d_distances);
#else
    lloydFun << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment, k, d, n, d_closest2, d_countDistances);
#endif
    changeAss << <numBlocksC, blockSizeC >> > (x->d_data, d_assignment, d_closest2, d_clusterSize, sumNewCenters[0]->d_data, d, nC, 0);
    
    cudaMemcpy(d_converged, &converged, 1 * sizeof(bool), cudaMemcpyHostToDevice);
    elkanMoveCenter << <numBlocksM, blockSizeM >> > (d_centerMovement, d_clusterSize, centers->d_data, sumNewCenters[0]->d_data, d_converged, k, d, nM);
    cudaMemcpy(&converged, d_converged, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
} 