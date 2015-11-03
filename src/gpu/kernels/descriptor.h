#ifndef __DESCRIPTOR_H__
#define __DESCRIPTOR_H__

#include "cuda_runtime.h"

//!
//! \brief Compute SIFT descriptors of passed SIFT keypoints
//! \param key_pts SIFT keypoints
//! \param orients keypoint orientations
//! \param grad keypoint gradients
//! \param num_pts number of keypoints
//! \param octave_width
//! \param octave_height
//! \param num_dogs number of difference of Gaussians in an
//! octave of the scale-space pyramid
//! \param xper level of scaling (i.e. as a result of
//! downsampling)
//! \param desc keypoint descriptors
//! \param x keypoint x coordinates (shallow-copy from \c key_pts
//! for easy access)
//! \param y keypoint y coordinates (shallow-copy from \c key_pts
//! for easy access)
//! \param stream
//!
void compute_sift_descriptors(const float4 * key_pts, const float2 * orients, const float2 * grad,
                              const int num_pts,
                              const int octave_width, const int octave_height,
                              const int num_dogs, const float xper,
                              float * desc, float * x, float * y,
                              cudaStream_t stream = 0);

#endif
