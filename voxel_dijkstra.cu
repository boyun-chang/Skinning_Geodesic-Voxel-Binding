#include "Voxel_dijkstra.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using namespace std;

#define INF 1e10f
#define BLOCK_SIZE 256

// 3D coordinate -> change 1D index
#define IDX(x, y, z, w, h) ((z) * (w) * (h) + (y) * (w) + (x))

// [kernel 1] reset
// Set distance 0 for Skeleton Node, infinite for remain
__global__ void initKernel(int* types, double* dists, int* parents, int totalSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int type = types[idx];
    if (type == SKELETON) {
        dists[idx] = 0.0f;
        parents[idx] = -1; // start point
    }
    else if (type == AIR) {
        dists[idx] = INF;
        parents[idx] = -2; // avoid
    }
    else {
        dists[idx] = INF; // Not visited yet
        parents[idx] = -1;
    }
}

// [kernel 2  Relaxation
// Check 6-way neighbors and update distances
__global__ void updateKernel(int* types, double* dists, int* parents, bool* changed,
    int width, int height, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = width * height * depth;

    if (idx >= totalSize) return;

    // AIR does not count
    if (types[idx] == AIR) return;

    // Current coordinate inversion
    
    int z = idx / (width * height);
    int rem = idx % (width * height);
    int y = rem / width;
    int x = rem % width;

    double myDist = dists[idx];
    double newDist = myDist;
    int bestParent = parents[idx];

    // 6-way neighbor offset
    int dx[6] = { -1, 1, 0, 0, 0, 0 };
    int dy[6] = { 0, 0, -1, 1, 0, 0 };
    int dz[6] = { 0, 0, 0, 0, -1, 1 };

    // Pull method: Look at your neighbors and see if there is a shorter route to you.
    for (int i = 0; i < 6; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        int nz = z + dz[i];

        // Check map range
        if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
            int nIdx = IDX(nx, ny, nz, width, height);

            // Check if the neighbor is valid (must not be AIR)
            if (types[nIdx] != AIR) {
                double neighborDist = dists[nIdx];

                // Weights are assumed to be 1.0 (received as an array if necessary)
                // If neighborDist is INF, the conditional statement does not pass because it is still INF even if added.
                if (neighborDist + 1.0f < newDist) {
                    newDist = neighborDist + 1.0f;
                    bestParent = nIdx;
                }
            }
        }
    }

    // If an update occurs, save the value and set a flag.
    if (newDist < myDist) {
        dists[idx] = newDist;
        parents[idx] = bestParent;
        *changed = true; // "Value changed" -> loop needs to be run again
    }
}

// Wrapper function called from the host (CPU)
extern "C" void runCUDADijkstra(
    const vector<int>& h_types,
    vector<double>& h_dists,
    vector<int>& h_parents,
    int w, int h, int d
) {
    int totalSize = w * h * d;
    size_t sizeInt = totalSize * sizeof(int);
    size_t sizeDouble = totalSize * sizeof(double);

    // Device memory pointer
    int* d_types, * d_parents;
    double* d_dists;
    bool* d_changed; // 수렴 여부 확인용

    // 1. Memory allocation
    cudaMalloc(&d_types, sizeInt);
    cudaMalloc(&d_dists, sizeDouble);
    cudaMalloc(&d_parents, sizeInt);
    cudaMalloc(&d_changed, sizeof(bool));

    // 2. Copy input data (Host -> Device)
    cudaMemcpy(d_types, h_types.data(), sizeInt, cudaMemcpyHostToDevice);

    // 3. Kernel execution settings
    int gridSize = (totalSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 3-1. Running the initialization kernel
    initKernel << <gridSize, BLOCK_SIZE >> > (d_types, d_dists, d_parents, totalSize);
    cudaDeviceSynchronize();

    // 3-2. Repeated update loop (Main Loop)
    int iter = 0;
    bool h_changed = false;

    do {
        h_changed = false;
        // Initialize the change flag to false and send it to the GPU.
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        // Run the update kernel
        updateKernel << <gridSize, BLOCK_SIZE >> > (d_types, d_dists, d_parents, d_changed, w, h, d);
        cudaDeviceSynchronize();

        // Check if there was a change (GPU -> CPU)
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        iter++;

        // Prevent infinite loops
        if (iter > totalSize) break;

    } while (h_changed);

    //printf("[GPU] Converged in %d iterations.\n", iter);

    // 4. Copy the result data (Device -> Host)
    h_dists.resize(totalSize);
    h_parents.resize(totalSize);
    cudaMemcpy(h_dists.data(), d_dists, sizeDouble, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_parents.data(), d_parents, sizeInt, cudaMemcpyDeviceToHost);

    // 5. Free memory
    cudaFree(d_types);
    cudaFree(d_dists);
    cudaFree(d_parents);
    cudaFree(d_changed);
}
