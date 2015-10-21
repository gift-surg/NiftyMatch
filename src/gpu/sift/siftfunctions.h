#ifndef __SIFTFUNCTIONS_H__
#define __SIFTFUNCTIONS_H__

#include "siftdata.h"
#include "pyramidata.h"
#include "siftparams.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

//! Compute the correspondencing SIFT features for A in B.
void compute_sift_matches(SiftData * A, SiftData * B,
                          float * distance, float ambiguity = 0.8f,
                          cudaStream_t stream = 0);

void compute_dog(PyramidData & pydata,
                 const int octave_width, const int octave_height,
                 cudaStream_t stream = 0);

void compute_gradients(PyramidData & pydata, const SiftParams & params,
                       const int octave_width, const int octave_height,
                       cudaStream_t stream = 0);

void compute_keypoints(PyramidData & pydata, const SiftParams & params,
                       const int octave, const int octave_width, const int octave_height,
                       cudaStream_t stream = 0);

void compute_keypoints_with_mask(PyramidData & pydata, SiftParams & params,
                                 cudaTextureObject_t mask,
                                 const int octave, const int octave_width, const int octave_height,
                                 cudaStream_t stream = 0);

void compute_orientations(PyramidData & pydata, const SiftParams & params,
                          const int octave, const int octave_width, const int octave_height,
                          cudaStream_t stream = 0);

void compute_descriptors(PyramidData & pydata, const SiftParams & params,
                         const int octave, const int octave_width, const int octave_height,
                         SiftData & data,
                         cudaStream_t stream = 0);

#endif
