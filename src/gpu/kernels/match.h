#ifndef __MATCH_H__
#define __MATCH_H__

#include <cuda.h>
#include <cuda_runtime.h>

//!
//! \brief Compute all mutual distances between SIFT
//! vectors in \c A and \c B
//! \param A
//! \param size_A
//! \param B
//! \param size_B
//! \param sift_vector_size
//! \param result distance values saved here
//! \param stream
//!
template<typename TYPE>
void compute_brute_force_distance(const TYPE * A, const int size_A,
                                  const TYPE * B, const int size_B,
                                  const int sift_vector_size,
                                  TYPE * result,
                                  cudaStream_t stream = 0);

//!
//! \brief Compute SIFT matches based on \c distance matrix
//! comparison using an \c ambiguity value. Exemplary use:
//! \code
//! get_sift_matches<float>(distance, A_size, B_size, B_size,
//!                         result, ambiguity, stream);
//! \endcode
//! \param distance
//! \param rows \c distance matrix row count
//! \param cols \c distance matrix column count
//! \param buffer_width
//! \param result match indices saved here
//! \param ambiguity
//! \param stream
//!
template<typename TYPE>
void get_sift_matches(const TYPE * distance,
                      const int rows, const int cols,
                      const int buffer_width,
                      int * result,
                      float ambiguity = 0.8f,
                      cudaStream_t stream = 0);

#endif
