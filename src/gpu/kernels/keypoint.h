#ifndef __KEYPOINT_H__
#define __KEYPOINT_H__

#include <cuda_runtime_api.h>

//!
//! \brief Find keypoints
//! \param current current Gaussian-smoothed image
//! \param down lower Gaussian-smoothed image
//! \param up higher Gaussian-smoothed image
//! \param width
//! \param height
//! \param peak_threshold peak threshold value for
//! sub-pixel refinement
//! \param edge_threshold edge threshold value for
//! sub-pixel refinement
//! \param xper level of scaling (i.e. as a result of
//! downsampling)
//! \param sigma_0 base sigma for Gaussians
//! \param num_dogs number of difference of Gaussians
//! \param dog current difference of Gaussian
//! \param result keypoints saved here
//! \param stream
//!
void find_keypoints(cudaTextureObject_t current,
                    cudaTextureObject_t down, cudaTextureObject_t up,
                    const int width, const int height,
                    const float peak_threshold, const float edge_threshold,
                    const float xper, const float sigma_0,
                    const int num_dogs, const int dog,
                    float4 * result,
                    cudaStream_t stream=0);

//!
//! \brief Find keypoints after applying \c mask. See unmasked version
//! for parameter documentation
//! \param current
//! \param mask
//! \param down
//! \param up
//! \param width
//! \param height
//! \param peak_threshold
//! \param edge_threshold
//! \param xper
//! \param sigma_0
//! \param num_dogs
//! \param dog
//! \param result
//! \param stream
//!
void find_keypoints(cudaTextureObject_t current, cudaTextureObject_t mask,
                    cudaTextureObject_t down, cudaTextureObject_t up,
                    const int width, const int height,
                    const float peak_threshold, const float edge_threshold,
                    const float xper, const float sigma_0,
                    const int num_dogs, const int dog,
                    float4 * result,
                    cudaStream_t stream=0);

#endif
