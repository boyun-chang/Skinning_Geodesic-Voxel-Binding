#pragma once
#include <vector>

using namespace std;

// Voxel type definition
enum VoxelType {
    AIR = 0,            // Empty space (excluding calculations)
    NORMAL = 1,         // General voxel (external/internal)
    SKELETON = 2        // Start Point (Skeleton)
};

// CUDA function wrapper
extern "C" void runCUDADijkstra(
    const vector<int>& h_voxelTypes,    // Input: Voxel type (1D array)
    vector<double>& h_distances,        // Output: Calculated distance
    vector<int>& h_parents,             // Output: Parent index for path traversal
    int width, int height, int depth    // Map size
);
