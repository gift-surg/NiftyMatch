#ifndef __KEYPOINT_H__
#define __KEYPOINT_H__

#include <cuda_runtime_api.h>

// version without a mask
void find_keypoints(cudaTextureObject_t current,
                    cudaTextureObject_t down, cudaTextureObject_t up,
                    const int width, const int height,
                    const float peak_thresh, const float edge_threshold,
                    const float xper, const float sigma_0,
                    const int num_dogs, const int dog,
                    float4 * result,
                    cudaStream_t stream=0);

// version with a mask
void find_keypoints(cudaTextureObject_t current, cudaTextureObject_t mask,
                    cudaTextureObject_t down, cudaTextureObject_t up,
                    const int width, const int height,
                    const float peak_thresh, const float edge_threshold,
                    const float xper, const float sigma_0,
                    const int num_dogs, const int dog,
                    float4 * result,
                    cudaStream_t stream=0);

#endif
