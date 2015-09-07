#ifndef __RANSAC_H__
#define __RANSAC_H__

#include <cuda_runtime_api.h>

#include<thrust/host_vector.h>

void align_points(const float* src_x, const float* src_y, const float* dst_x, const float* dst_y,
                  float* c_src_x, float* c_src_y, float* c_dst_x, float* c_dst_y,
                  const int * matches, const int num_pts, cudaStream_t stream=0);

bool ransac_homography(float* src_x, float* src_y, float* dst_x, float* dst_y,
                       const int src_size, const int dst_size, float inlier_threshold,
                       int iterations, float *homography, cudaStream_t stream=0);

bool ransac_translation(float* src_x, float* src_y, float* dst_x, float* dst_y,
                        const int src_size, const int dst_size, float inlier_threshold,
                        int iterations, float *homography, cudaStream_t stream=0);

#endif
