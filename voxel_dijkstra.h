#pragma once
#include <vector>

using namespace std;

// 복셀 타입 정의
enum VoxelType {
    AIR = 0,            // 빈 공간 (계산 제외)
    NORMAL = 1,         // 일반 복셀 (외/내부)
    SKELETON = 2        // 시작점 (스켈레톤)
};

// CUDA 함수 래퍼 
extern "C" void runCUDADijkstra(
    const vector<int>& h_voxelTypes,   // 입력: 복셀 타입 (1D 배열)
    vector<double>& h_distances,        // 출력: 계산된 거리
    vector<int>& h_parents,            // 출력: 경로 역추적용 부모 인덱스
    int width, int height, int depth        // 맵 크기
);