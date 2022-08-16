#pragma once

#include "Dataset.h"
#include <fstream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline Dataset* loadDataset(std::string const& filename, int n, int d) {
    std::string number;
    Dataset* x = nullptr;
    std::fstream file;

    file.open(filename, std::ios::in);
    x = new Dataset(n, d);
    int i = 0;
    while (getline(file, number, ','))
    {
        x->data[i] = atof(number.c_str());
        i++;
        if (i >= n * d)
            break;
    }
    file.close();
    return x;
}

inline Dataset* init_centers(Dataset const& x, unsigned short k) {
    //srand(time(NULL));
    int* chosen_pts = new int[k];
    Dataset* c = new Dataset(k, x.d);
    for (int i = 0; i < k; ++i) {
        bool acceptable = true;
        do {
            acceptable = true;
            auto ran = rand() % x.n;
            chosen_pts[i] = ran;
            for (int j = 0; j < i; ++j) {
                if (chosen_pts[i] == chosen_pts[j]) {
                    acceptable = false;
                    break;
                }
            }
        } while (!acceptable);

        double* cdp = c->data + i * x.d;
        memcpy(cdp, x.data + chosen_pts[i] * x.d, sizeof(double) * x.d);
    }

    delete[] chosen_pts;
    return c;
}

inline void assign(Dataset const& x, Dataset const& c, unsigned short* assignment) {
    for (int i = 0; i < x.n; ++i) {
        double shortestDist2 = std::numeric_limits<double>::max();
        int closest = 0;
        for (int j = 0; j < c.n; ++j) {
            double d2 = 0.0, * a = x.data + i * x.d, * b = c.data + j * x.d;
            for (; a != x.data + (i + 1) * x.d; ++a, ++b) {
                d2 += (*a - *b) * (*a - *b);
            }
            if (d2 < shortestDist2) {
                shortestDist2 = d2;
                closest = j;
            }
        }
        assignment[i] = closest;
        //assignment[i] = c.n +1;
    }
}

inline void addVectors(double* a, double const* b, int d) {
    double const* end = a + d;
    while (a < end) {
        *(a++) += *(b++);
    }
}

inline void subVectors(double* a, double const* b, int d) {
    double const* end = a + d;
    while (a < end) {
        *(a++) -= *(b++);
    }
}