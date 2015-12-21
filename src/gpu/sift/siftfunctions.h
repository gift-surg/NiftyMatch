#ifndef __SIFTFUNCTIONS_H__
#define __SIFTFUNCTIONS_H__

#include "siftdata.h"
#include "pyramidata.h"
#include "siftparams.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

//!
//! \brief Compute corresponding SIFT features for \c A in \c B
//! \param A
//! \param B
//! \param distance matrix of mutual distances between keypoints
//! \param ambiguity epsilon value in comparing distances for matching
//! \param stream
//!
void compute_sift_matches(SiftData * A, SiftData * B,
                          float * distance, float ambiguity = 0.8f,
                          cudaStream_t stream = 0);

//!
//! \brief Compute difference of Gaussians for \c octave
//! \param pydata differences saved here
//! \param octave_width
//! \param octave_height
//! \param stream
//!
void compute_dog(PyramidData & pydata,
                 const int octave_width, const int octave_height,
                 cudaStream_t stream = 0);

//!
//! \brief Compute gradients within all octaves
//! \param pydata gradients saved here
//! \param params
//! \param octave_width
//! \param octave_height
//! \param stream
//!
void compute_gradients(PyramidData & pydata, const SiftParams & params,
                       const int octave_width, const int octave_height,
                       cudaStream_t stream = 0);

//!
//! \brief Compute keypoints for specified \c octave
//! \param pydata keypoints saved here
//! \param params
//! \param octave
//! \param octave_width
//! \param octave_height
//! \param stream
//!
void compute_keypoints(PyramidData & pydata, const SiftParams & params,
                       const int octave, const int octave_width, const int octave_height,
                       cudaStream_t stream = 0);

//!
//! \brief Compute keypoints for specified \c octave, after masking image with \c mask
//! \param pydata
//! \param params
//! \param mask
//! \param octave
//! \param octave_width
//! \param octave_height
//! \param stream
//! \sa compute_keypoints
//!
void compute_keypoints_with_mask(PyramidData & pydata, SiftParams & params,
                                 cudaTextureObject_t mask,
                                 const int octave, const int octave_width, const int octave_height,
                                 cudaStream_t stream = 0);

//!
//! \brief Compute SIFT orientations of keypoints for passed \c octave
//! \param pydata orientations saved here
//! \param params
//! \param octave
//! \param octave_width
//! \param octave_height
//! \param stream
//!
void compute_orientations(PyramidData & pydata, const SiftParams & params,
                          const int octave, const int octave_width, const int octave_height,
                          cudaStream_t stream = 0);

//!
//! \brief Compute descriptors for passed \c octave
//! \param pydata provides keypoints, orientations and gradients
//! \param params
//! \param octave
//! \param octave_width
//! \param octave_height
//! \param data descriptors saved here
//! \param stream
//!
void compute_descriptors(PyramidData & pydata, const SiftParams & params,
                         const int octave, const int octave_width, const int octave_height,
                         SiftData & data,
                         cudaStream_t stream = 0);

#endif
