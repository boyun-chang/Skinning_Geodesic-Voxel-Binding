#include "Voxel_dijkstra.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using namespace std;

#define INF 1e10f
#define BLOCK_SIZE 256

// 3D 좌표 -> 1D 인덱스 변환
#define IDX(x, y, z, w, h) ((z) * (w) * (h) + (y) * (w) + (x))
//#define IDX(x, y, z, w, h) ((x) * (w) * (h) + (y) * (w) + (z))

// [커널 1] 초기화
// 스켈레톤 노드는 거리 0, 나머지는 무한대로 설정
__global__ void initKernel(int* types, double* dists, int* parents, int totalSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int type = types[idx];
    if (type == SKELETON) {
        dists[idx] = 0.0f;
        parents[idx] = -1; // 시작점
    }
    else if (type == AIR) {
        dists[idx] = INF;
        parents[idx] = -2; // 무효
    }
    else {
        dists[idx] = INF; // 아직 방문 안 함
        parents[idx] = -1;
    }
}

// [커널 2] 거리 갱신 (Relaxation)
// 6방향 이웃을 확인하고 거리를 갱신
__global__ void updateKernel(int* types, double* dists, int* parents, bool* changed,
    int width, int height, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = width * height * depth;

    if (idx >= totalSize) return;

    // AIR는 계산하지 않음
    if (types[idx] == AIR) return;

    // 현재 좌표 역산
    
    int z = idx / (width * height);
    int rem = idx % (width * height);
    int y = rem / width;
    int x = rem % width;
    /*
    int x = idx / (width * height);
    int rem = idx % (width * height);
    int y = rem / width;
    int z = rem % width;*/

    double myDist = dists[idx];
    double newDist = myDist;
    int bestParent = parents[idx];

    // 6방향 이웃 오프셋
    int dx[6] = { -1, 1, 0, 0, 0, 0 };
    int dy[6] = { 0, 0, -1, 1, 0, 0 };
    int dz[6] = { 0, 0, 0, 0, -1, 1 };

    // Pull 방식: 내 주변 이웃을 보고, 나에게 오는 더 짧은 길이 있는지 확인
    for (int i = 0; i < 6; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        int nz = z + dz[i];

        // 맵 범위 체크
        if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
            int nIdx = IDX(nx, ny, nz, width, height);

            // 이웃이 유효한지 확인 (AIR가 아니어야 함)
            if (types[nIdx] != AIR) {
                double neighborDist = dists[nIdx];

                // 가중치는 1.0으로 가정 (필요시 배열로 받음)
                // 만약 neighborDist가 INF라면 더해도 INF이므로 조건문 통과 못함
                if (neighborDist + 1.0f < newDist) {
                    newDist = neighborDist + 1.0f;
                    bestParent = nIdx;
                }
            }
        }
    }

    // 갱신이 발생했으면 값 저장 및 플래그 설정
    if (newDist < myDist) {
        dists[idx] = newDist;
        parents[idx] = bestParent;
        *changed = true; // "값이 바뀌었음" -> 루프를 한 번 더 돌려야 함
    }
}

// 호스트(CPU)에서 호출하는 래퍼 함수
extern "C" void runCUDADijkstra(
    const vector<int>& h_types,
    vector<double>& h_dists,
    vector<int>& h_parents,
    int w, int h, int d
) {
    int totalSize = w * h * d;
    size_t sizeInt = totalSize * sizeof(int);
    size_t sizeDouble = totalSize * sizeof(double);

    // 디바이스 메모리 포인터
    int* d_types, * d_parents;
    double* d_dists;
    bool* d_changed; // 수렴 여부 확인용

    // 1. 메모리 할당
    cudaMalloc(&d_types, sizeInt);
    cudaMalloc(&d_dists, sizeDouble);
    cudaMalloc(&d_parents, sizeInt);
    cudaMalloc(&d_changed, sizeof(bool));

    // 2. 입력 데이터 복사 (Host -> Device)
    cudaMemcpy(d_types, h_types.data(), sizeInt, cudaMemcpyHostToDevice);

    // 3. 커널 실행 설정
    int gridSize = (totalSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 3-1. 초기화 커널 실행
    initKernel << <gridSize, BLOCK_SIZE >> > (d_types, d_dists, d_parents, totalSize);
    cudaDeviceSynchronize();

    // 3-2. 반복 갱신 루프 (Main Loop)
    int iter = 0;
    bool h_changed = false;

    do {
        h_changed = false;
        // 변경 플래그를 false로 초기화 후 GPU로 전송
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        // 업데이트 커널 실행
        updateKernel << <gridSize, BLOCK_SIZE >> > (d_types, d_dists, d_parents, d_changed, w, h, d);
        cudaDeviceSynchronize();

        // 변경 사항이 있었는지 확인 (GPU -> CPU)
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        iter++;

        // 무한 루프 방지
        if (iter > totalSize) break;

    } while (h_changed);

    //printf("[GPU] Converged in %d iterations.\n", iter);

    // 4. 결과 데이터 복사 (Device -> Host)
    h_dists.resize(totalSize);
    h_parents.resize(totalSize);
    cudaMemcpy(h_dists.data(), d_dists, sizeDouble, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_parents.data(), d_parents, sizeInt, cudaMemcpyDeviceToHost);

    // 5. 메모리 해제
    cudaFree(d_types);
    cudaFree(d_dists);
    cudaFree(d_parents);
    cudaFree(d_changed);
}