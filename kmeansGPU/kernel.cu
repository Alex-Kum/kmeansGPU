//https://github.com/Alex-Kum/kmeansGPU


//#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpufunctions.cuh"
#include "lloyd.h"
#include "elkan_kmean.h"
#include "hamerly.h"
#include "fbelkan.h"
#include "moelkan.h"
#include "fbhamerly.h"

#include <stdio.h>
#include <chrono>
#include <vector>

using namespace std;;

int main(){
    Dataset* x = loadDataset("KEGGNetwork_clean.data", 65554, 28);
    //Dataset* x = loadDataset("USCensus_clean.data", 2458285, 68);
    //Dataset* x = loadDataset("gassensor_clean.data", 13910, 128);

    cout << "Dataset loaded" << endl;

    vector< std::chrono::duration<double>> init;
    vector< std::chrono::duration<double>> results;

    int k = 4;
    int repeat = 3;
    cout << "k: " << k << endl;
    auto start1 = std::chrono::system_clock::now();
    Dataset* initialCenters = init_centers(*x, k);
    auto end1 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds1 = end1 - start1;
    std::cout << "Sekunden init center: " << elapsed_seconds1.count() << "\n";

    for (int i = 0; i < repeat; i++) {
        auto* alg = new LloydKmeans();
        //auto* alg = new ElkanKmeans();
        //auto* alg = new FBElkanKmeans(); 
        //auto* alg = new MOElkanKmeans();
        //* alg = new HamerlyKmeans();
        //auto* alg = new FBHamerly();

        std::cout << "Alg: " << alg->getName() << std::endl;
        unsigned short* assignment = new unsigned short[x->n];

        auto start2 = std::chrono::system_clock::now();
        assign(*x, *initialCenters, assignment);
        auto end2 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
        std::cout << "Sekunden initassign: " << elapsed_seconds2.count() << "\n";

            
        auto duration = alg->initialize(x, k, assignment, 1);
        init.push_back(duration);

        auto start = std::chrono::system_clock::now();
        std::cout << "alg run start" << std::endl;
        int iterations = alg->run(500);
        std::cout << "alg run end" << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        results.push_back(elapsed_seconds);
        std::cout << "Sekunden: " << elapsed_seconds.count() << "\n";

        delete[] assignment;
        delete alg;
    }
    delete initialCenters;
    cudaDeviceSynchronize();

    auto sumInit = 0.0;
    for (auto& e : init) {
        cout << e.count() << ", ";
        sumInit += e.count();
    }
    cout << "               Init Avg:" << sumInit/repeat << ", ";
    cout << endl;

    int counter = 0;
    auto sum = 0.0;
    for (auto& e : results) {
        cout << e.count() << ", ";
        sum += e.count();
        counter++;
    }
    cout << "               Total Avg:" << sum/repeat << ", ";
    cout << endl;

    delete x;

    auto cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}