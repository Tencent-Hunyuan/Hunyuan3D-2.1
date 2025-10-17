#ifndef RASTERIZER_H_
#define RASTERIZER_H_

#include <torch/extension.h>
#include <vector>

// Only include Metal headers when compiling as Objective-C++
#ifdef __OBJC__
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#define INT64 unsigned long long
#define MAXINT 2147483647

// Use conditional compilation for Objective-C specific code
#ifdef __OBJC__
#define HOST_DEVICE __attribute__((__host__, __device__))
extern "C" {
    id<MTLDevice> getMetalDevice(); // Declare function to get device
}
#else
#define HOST_DEVICE
#endif

HOST_DEVICE inline float calculateSignedArea2(float* a, float* b, float* c) {
    return ((c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1]));
}

HOST_DEVICE inline void calculateBarycentricCoordinate(float* a, float* b, float* c, float* p,
    float* barycentric)
{
    float beta_tri = calculateSignedArea2(a, p, c);
    float gamma_tri = calculateSignedArea2(a, b, p);
    float area = calculateSignedArea2(a, b, c);
    if (area == 0) {
        barycentric[0] = -1.0;
        barycentric[1] = -1.0;
        barycentric[2] = -1.0;
        return;
    }
    float tri_inv = 1.0 / area;
    float beta = beta_tri * tri_inv;
    float gamma = gamma_tri * tri_inv;
    float alpha = 1.0 - beta - gamma;
    barycentric[0] = alpha;
    barycentric[1] = beta;
    barycentric[2] = gamma;
}

HOST_DEVICE inline bool isBarycentricCoordInBounds(float* barycentricCoord) {
    return barycentricCoord[0] >= 0.0 && barycentricCoord[0] <= 1.0 &&
           barycentricCoord[1] >= 0.0 && barycentricCoord[1] <= 1.0 &&
           barycentricCoord[2] >= 0.0 && barycentricCoord[2] <= 1.0;
}


std::vector<std::vector<torch::Tensor>> build_hierarchy(std::vector<torch::Tensor> view_layer_positions, std::vector<torch::Tensor> view_layer_normals, int num_level, int resolution);

std::vector<std::vector<torch::Tensor>> build_hierarchy_with_feat(
    std::vector<torch::Tensor> view_layer_positions,
    std::vector<torch::Tensor> view_layer_normals,
    std::vector<torch::Tensor> view_layer_feats,
    int num_level, int resolution);

#endif
