#pragma once
#include <stdio.h>
#include "cuda_runtime.h"

#define _HUGE_ENUF  1e+300
#define INFINITY   ((float)(_HUGE_ENUF * _HUGE_ENUF))
#define DISTANCES 0

static __global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

inline __device__ double atomicMin_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

inline __device__ double atomicMax_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

inline __device__ double distDD(const double* data, int x, int y, int dim) {
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        double diff = data[x * dim + i] - data[y * dim + i];
        result += diff * diff;
    }
    return result;
}

inline __device__ double dist(const double* data, const double* center, int x, int y, int dim) {
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        double diff = data[x * dim + i] - center[y * dim + i];
        result += diff * diff;
    }
    return result;
}

inline __device__ void addVectorsAtomic(double* a, double const* b, int d) {
    double const* end = a + d;
    while (a < end) {
        //*(a++) += *(b++);
        double bVal = *(b++);
        atomicAdd(a, bVal);
        a++;
    }
}

inline __device__ void subVectorsAtomic(double* a, double const* b, int d) {
    double const* end = a + d;
    while (a < end) {
        double bVal = *(b++);
        atomicAdd(a, -bVal);
        a++;
    }
}

static __global__ void elkanMoveCenter(double* centerMovement, int* clusterSize, double* center, double* sumNewCenters, bool* converged, int k, int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        //printf("move\n");
        centerMovement[i] = 0.0;
        int totalClusterSize = clusterSize[i];

        if (totalClusterSize > 0) {
            for (int d = 0; d < dim; ++d) {
                double z = 0.0;
                z += sumNewCenters[i * dim + d];
                z /= totalClusterSize;
                centerMovement[i] += (z - center[i * dim + d]) * (z - center[i * dim + d]);
                center[i * dim + d] = z;
            }
        }

        centerMovement[i] = sqrt(centerMovement[i]);
        if (centerMovement[i] > 0) {
            *converged = false;
        }
    }
}

static __global__ void elkanMoveCenterFB(double* centerMovement, int* clusterSize, double* center, double* sumNewCenters, double* oldcenters, bool* converged, int k, int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        centerMovement[i] = 0.0;
        int totalClusterSize = clusterSize[i];

        if (totalClusterSize > 0) {
            for (int d = 0; d < dim; ++d) {
                double z = 0.0;
                z += sumNewCenters[i * dim + d];
                z /= totalClusterSize;
                centerMovement[i] += (z - center[i * dim + d]) * (z - center[i * dim + d]);
                oldcenters[i * dim + d] = center[i * dim + d];
                center[i * dim + d] = z;
            }
        }
        centerMovement[i] = sqrt(centerMovement[i]);

        if (centerMovement[i] > 0)
            *converged = false;
    }
}

static __global__ void elkanFBMoveAddition(double* oldcenters, double* oldcenter2newcenterDis, double* center, int d, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / n;
    int c2 = i % n;

    if (c1 != c2 && i < n * n) {
        for (int dim = 0; dim < d; ++dim) {
            oldcenter2newcenterDis[c1 * k + c2] += (oldcenters[c1 * d + dim] - center[c2 * d + dim]) * (oldcenters[c1 * d + dim] - center[c2 * d + dim]);
        }
        oldcenter2newcenterDis[c1 * k + c2] = sqrt(oldcenter2newcenterDis[c1 * k + c2]);
    }
}

static __global__ void changeAss(double* data, unsigned short* assignment, unsigned short* closest2, int* clusterSize, double* sumNewCenters, int dim, int n, int offset) {
    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    //printf("changeass \n");
    if (i < n) {
        if (assignment[i] != closest2[i]) {
            unsigned short oldAssignment = assignment[i];

            atomicSub(&clusterSize[assignment[i]], 1);
            atomicAdd(&clusterSize[closest2[i]], 1);
            double* xp = data + i * dim;
            assignment[i] = closest2[i];

            subVectorsAtomic(sumNewCenters + oldAssignment * dim, xp, dim);
            addVectorsAtomic(sumNewCenters + closest2[i] * dim, xp, dim);
        }
    }
}

static __global__ void initArrDouble(double* arr, double val, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        arr[index] = val;
    }
}


static __global__ void innerProd(double* centerCenterDist, double* s, const double* data, int dim, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = index / n;
    int c2 = index % n;

    if (c1 != c2 && index < n * n) {
        //printf("inner\n");
        double distance = distDD(data, c1, c2, dim);
        centerCenterDist[index] = sqrt(distance) / 2.0;
        atomicMin_double(&s[c1], centerCenterDist[index]);
    }
}

static __global__ void innerProdMO(double* centerCenterDist, double* oldcenterCenterDistDiv2, double* s, const double* data, int dim, int k, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = index / n;
    int c2 = index % n;

    if (c1 != c2 && index < n * n) {
        double distance = distDD(data, c1, c2, dim);
        oldcenterCenterDistDiv2[c1 * k + c2] = centerCenterDist[c1 * k + c2];
        centerCenterDist[index] = sqrt(distance) / 2.0;
        atomicMin_double(&s[c1], centerCenterDist[index]);
    }
}

static __global__ void elkanFBMoveAdditionHam(double* oldcenters, double* oldcenter2newcenterDis, double* maxoldcenter2newcenterDis, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double maxCenterDis = INFINITY;
        for (int j = 0; j < k; j++) {
            if (oldcenter2newcenterDis[i * k + j] < maxCenterDis) {
                maxCenterDis = oldcenter2newcenterDis[i * k + j];
            }
        }
        maxoldcenter2newcenterDis[i] = maxCenterDis;
    }
}

static __global__ void updateBound(double* lower, double* upper, double* centerMovement, unsigned short* assignment, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        //printf("bound\n");
        upper[i] += centerMovement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * k + j] -= centerMovement[j];
        }
    }
}

static __global__ void updateBoundShared(double* lower, double* upper, double* centerMovement, unsigned short* assignment, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double movement[256];
    if (threadIdx.x < 256) {
        movement[threadIdx.x] = centerMovement[threadIdx.x];
    }
    __syncthreads();

    if (i < n) {
        upper[i] += movement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * k + j] -= movement[j];
        }
    }
}

static __global__ void updateBoundFBShared(double* lower, double* upper, double* ub_old, double* centerMovement, unsigned short* assignment, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double movement[256];
    if (threadIdx.x < 256) {
        movement[threadIdx.x] = centerMovement[threadIdx.x];
    }
    __syncthreads();

    if (i < n) {
        ub_old[i] = upper[i];
        upper[i] += movement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * k + j] -= movement[j];
        }
    }
}

static __global__ void updateBoundHam(double* lower, double* upper, double* centerMovement, unsigned short* assignment, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double maxMovement = 0;
        upper[i] += centerMovement[assignment[i]];

        for (int j = 0; j < k; ++j) {
            if (centerMovement[j] > maxMovement)
                maxMovement = centerMovement[j];
        }
        lower[i] -= maxMovement;
    }
}

static __global__ void updateBoundHamShared(double* lower, double* upper, double* centerMovement, unsigned short* assignment, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double movement[256];
    if (threadIdx.x < 256) {
        movement[threadIdx.x] = centerMovement[threadIdx.x];
    }
    __syncthreads();

    if (i < n) {
        double maxMovement = 0;
        upper[i] += movement[assignment[i]];

        for (int j = 0; j < k; ++j) {
            if (movement[j] > maxMovement)
                maxMovement = movement[j];
        }
        lower[i] -= maxMovement;
    }
}

static __global__ void updateBoundHamFBShared(double* lower, double* upper, double* ub_old, double* centerMovement, unsigned short* assignment, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double movement[256];
    if (threadIdx.x < 256) {
        movement[threadIdx.x] = centerMovement[threadIdx.x];
    }
    __syncthreads();

    if (i < n) {
        double maxMovement = 0;
        ub_old[i] = upper[i];
        upper[i] += movement[assignment[i]];

        for (int j = 0; j < k; ++j) {
            if (movement[j] > maxMovement)
                maxMovement = movement[j];
        }
        lower[i] -= maxMovement;
    }
}

static __global__ void updateBoundFB(double* lower, double* upper, double* ub_old, double* centerMovement, unsigned short* assignment, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ub_old[i] = upper[i];
        upper[i] += centerMovement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * k + j] -= centerMovement[j];
        }
    }
}

static __global__ void updateBoundFBHam(double* lower, double* upper, double* ub_old, double* centerMovement, unsigned short* assignment, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double maxMovement = 0;
        ub_old[i] = upper[i];
        upper[i] += centerMovement[assignment[i]];

        for (int j = 0; j < k; ++j) {
            if (j == assignment[i])
                continue;
            if (centerMovement[j] > maxMovement)
                maxMovement = centerMovement[j];
        }    
        lower[i] -= maxMovement;
    }
}
static __global__ void updateBoundMO(double* upper, double* ub_old, double* centerMovement, unsigned short* assignment, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ub_old[i] = upper[i];
        upper[i] += centerMovement[assignment[i]];
    }
}

static __global__ void lloydFun(double* data, double* center, unsigned short* assignment, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
#if DISTANCES
        unsigned long long int c;
#endif
        closest2[i] = assignment[i];
        unsigned short jjj = assignment[i];
        double closestDistance = INFINITY;
        for (int j = 0; j < k; ++j) {
            double curDistance = sqrt(dist(data, center, i, j, dim));
            // atomicAdd(countDistances, 1);
#if DISTANCES
            c = atomicAdd(countDistances, 1);
            if (c == 18446744073709551615) {
                printf("OVERFLOW");
            }
#endif
            if (curDistance < closestDistance) {
                closestDistance = curDistance;
                closest2[i] = j;
            }
        }
    }
}

static __global__ void lloydFunK(double* data, double* center, int k, int dim, int n, unsigned short* closest2, double* distances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int c2 = i % k;

    if (c1 < n) {
        distances[i] = sqrt(dist(data, center, c1, c2, dim));
    }
}

static __global__ void elkanFun(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, int offset, unsigned long long int* countDistances) {

    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
        bool r = true;
        double localUpper = upper[i];
        //printf("upper: %f\n", upper[i]);
#if DISTANCES
        unsigned long long int c;
#endif
        if (localUpper > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (localUpper <= lower[i * k + j]) { continue; }
                if (localUpper <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    localUpper = sqrt(dist(data, center, i, closest2[i], dim));
#if DISTANCES
                    c = atomicAdd(countDistances, 1);
                    if (c == 18446744073709551615) {
                        printf("OVERFLOW");
                    }
#endif
                    lower[i * k + closest2[i]] = localUpper;
                    r = false;
                    if ((localUpper <= lower[i * k + j]) || (localUpper <= centerCenterDistDiv2[closest2[i] * k + j])) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                lower[i * k + j] = sqrt(dist(data, center, i, j, dim));
#if DISTANCES
                c = atomicAdd(countDistances, 1);
                if (c == 18446744073709551615) {
                    printf("OVERFLOW");
                }
#endif
                if (lower[i * k + j] < localUpper) {
                    closest2[i] = j;
                    localUpper = lower[i * k + j];
                }
            }
            upper[i] = localUpper;
        }
    }
}

static __global__ void elkanFunK(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, bool* calculated, double* distances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int j = i % k;

    if (c1 < n) {
        closest2[c1] = assignment[c1];
        calculated[i] = upper[c1] > s[closest2[c1]] && upper[c1] >= lower[i] && upper[c1] >= centerCenterDistDiv2[closest2[c1] * k + j];
        if (calculated[i]) {
            lower[i] = sqrt(dist(data, center, c1, j, dim));
        }
    }
}

static __global__ void fbelkanFunK(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* oldcenter2newcenterDis, double* ub_old, bool* calculated, int n, int dim, int k, unsigned short* closest2, double* centerCenterDistDiv2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int j = i % k;

    if (c1 < n) {
        closest2[c1] = assignment[c1];
        calculated[i] = upper[c1] >= lower[i] && upper[c1] >= oldcenter2newcenterDis[assignment[c1] * k + j] - ub_old[c1]
            && upper[c1] >= centerCenterDistDiv2[closest2[c1] * k + j];
        if (calculated[i]) {
            lower[i] = sqrt(dist(data, center, c1, j, dim));
        }
    }
}

static __global__ void moelkanFunK(double* data, double* center, unsigned short* assignment, double* distance, double* upper,
    double* s, double* oldcenter2newcenterDis, double* ub_old, bool* calculated, int n, int dim, int k, unsigned short* closest2, 
    double* centerCenterDistDiv2, double* oldcenterCenterDistDiv2, double* centerMovement) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int j = i % k;

    if (c1 < n) {
        closest2[c1] = assignment[c1];
        calculated[i] = upper[c1] > s[closest2[c1]] && upper[c1] >= oldcenter2newcenterDis[assignment[c1] * k + j] - ub_old[c1]
            && 2.0 * (oldcenterCenterDistDiv2[assignment[c1] * k + j]) - ub_old[c1] - centerMovement[j] && upper[c1] >= centerCenterDistDiv2[closest2[c1] * k + j];
        if (calculated[i]) {
            distance[i] = sqrt(dist(data, center, c1, j, dim));
        }
    }
}

static __global__ void elkCombineK(double* lower, double* upper, int k, int n, unsigned short* closest2, bool* calculated, double* dist) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        for (int j = 0; j < k; j++) {
            if (calculated[i * k + j]) {
                if (lower[i * k + j] < upper[i]) {
                    closest2[i] = j;
                    upper[i] = lower[i * k + j];
                }
            }
        }
    }

}

static __global__ void hamerlyFun(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
#if DISTANCES
        unsigned long long int c;
#endif
        if (upper[i] >= s[closest2[i]] && upper[i] >= lower[i]) {
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = sqrt(dist(data, center, i, j, dim));
#if DISTANCES
                c = atomicAdd(countDistances, 1);
                if (c == 18446744073709551615) {
                    printf("OVERFLOW");
                }
#endif
                if (curDistance < closestDistance) {
                    secondClosestDist = closestDistance;
                    closestDistance = curDistance;
                    closest2[i] = j;
                }
                else if (curDistance < secondClosestDist) {
                    secondClosestDist = curDistance;
                }
            }
            upper[i] = closestDistance;
            lower[i] = secondClosestDist;
        }
    }
}


static __global__ void hamerlyFunK(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
         double* s, double* centerCenterDistDiv2,  int k, int dim, int n, unsigned short* closest2, double* distances, bool* calculated) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int c2 = i % k;

    if (c1 < n) {
        calculated[c1] = upper[c1] > s[assignment[c1]] && upper[c1] >= lower[c1]; 
        if (calculated[c1])
            distances[i] = sqrt(dist(data, center, c1, c2, dim));
    }
}

static __global__ void fbhamerlyFunK(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, double* maxoldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2, double* distances, bool* calculated) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int c2 = i % k;

    if (c1 < n) {
        calculated[c1] = upper[c1] > s[assignment[c1]] && upper[c1] >= lower[c1] && upper[c1] >= maxoldcenter2newcenterDis[assignment[c1]] - ub_old[c1];    
        if (calculated[c1])
            distances[i] = sqrt(dist(data, center, c1, c2, dim));
    }
}


static __global__ void hamCombineK(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
         double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, bool* calculated, double* distances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (calculated[i]) {
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = distances[i * k + j];
                if (curDistance < closestDistance) {
                    secondClosestDist = closestDistance;
                    closestDistance = curDistance;
                    closest2[i] = j;
                }
                else if (curDistance < secondClosestDist) {
                    secondClosestDist = curDistance;
                }
            }
            upper[i] = closestDistance;
            lower[i] = secondClosestDist;
        }
    }
}

static __global__ void lloydCombineK(double* data, double* center, int k, int n, unsigned short* closest2, double* distances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double closestDistance = INFINITY;
        for (int j = 0; j < k; ++j) {
            double curDistance = distances[i * k + j];
            if (curDistance < closestDistance) {              
                closestDistance = curDistance;
                closest2[i] = j;
            }          
        }
    }
}


static __global__ void elkanFunFB(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, double* oldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        closest2[i] = assignment[i];
        bool r = true;
#if DISTANCES
        unsigned long long int c;
#endif
        if (upper[i] > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (upper[i] <= lower[i * k + j]) { continue; }
                if (upper[i] <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) { continue; }
                if (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    upper[i] = sqrt(dist(data, center, i, closest2[i], dim));
#if DISTANCES
                    c = atomicAdd(countDistances, 1);
                    if (c == 18446744073709551615) {
                        printf("OVERFLOW");
                    }
#endif
                    lower[i * k + closest2[i]] = upper[i];
                    r = false;
                    if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j]) || upper[i] <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                lower[i * k + j] = sqrt(dist(data, center, i, j, dim));
#if DISTANCES
                c = atomicAdd(countDistances, 1);
                if (c == 18446744073709551615) {
                    printf("OVERFLOW");
                }
#endif
                if (lower[i * k + j] < upper[i]) {
                    closest2[i] = j;
                    upper[i] = lower[i * k + j];
                }
            }
        }
    }
}

static __global__ void elkanFunMO(double* data, double* center, unsigned short* assignment, double* upper,
    double* s, double* centerCenterDistDiv2, double* oldcenter2newcenterDis, double* oldcenterCenterDistDiv2, double* ub_old, double* centerMovement, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        closest2[i] = assignment[i];
        double localUpper = upper[i];
        bool r = true;
        unsigned long long int c;

        if (localUpper > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (localUpper <= 2.0 * (oldcenterCenterDistDiv2[assignment[i] * k + j]) - ub_old[i] - centerMovement[j]) { continue; }
                if (localUpper <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) { continue; }
                if (localUpper <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    localUpper = sqrt(dist(data, center, i, closest2[i], dim));
#if DISTANCES
                    c = atomicAdd(countDistances, 1);
                    if (c == 18446744073709551615) {
                        printf("OVERFLOW");
                    }
#endif
                    r = false;
                    if ((localUpper <= 2.0 * (oldcenterCenterDistDiv2[assignment[i] * k + j]) - ub_old[i] - centerMovement[j]) || (localUpper <= centerCenterDistDiv2[closest2[i] * k + j])) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                double inner = sqrt(dist(data, center, i, j, dim));
#if DISTANCES
                c = atomicAdd(countDistances, 1);
                if (c == 18446744073709551615) {
                    printf("OVERFLOW");
                }
#endif
                if (inner < localUpper) {
                    closest2[i] = j;
                    localUpper = inner;
                }
            }
        }
        upper[i] = localUpper;
    }
}

static __global__ void elkanFunFBHam(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, double* maxoldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        closest2[i] = assignment[i];
#if DISTANCES
        unsigned long long int c;
#endif
        if (upper[i] > s[closest2[i]] && upper[i] >= lower[i] && upper[i] >= maxoldcenter2newcenterDis[assignment[i]] - ub_old[i]) {
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = sqrt(dist(data, center, i, j, dim));
#if DISTANCES
                c = atomicAdd(countDistances, 1);
                if (c == 18446744073709551615) {
                    printf("OVERFLOW");
                }
#endif
                if (curDistance < closestDistance) {
                    secondClosestDist = closestDistance;
                    closestDistance = curDistance;
                    closest2[i] = j;
                }
                else if (curDistance < secondClosestDist) {
                    secondClosestDist = curDistance;
                }
            }
            upper[i] = closestDistance;
            lower[i] = secondClosestDist;
        }
    }
}