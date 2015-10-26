#ifndef __MATCH_H__
#define __MATCH_H__

#include <cuda.h>
#include <cuda_runtime.h>

template<typename TYPE>
void compute_brute_force_distance(const TYPE * A, const int size_A,
                                  const TYPE * B, const int size_B,
                                  const int sift_vector_size,
                                  TYPE * result,
                                  cudaStream_t stream = 0);

template<typename TYPE>
void get_sift_matches(TYPE * distance,
                      const int rows, const int cols,
                      const int buffer_width,
                      int * result,
                      float ambiguity = 0.8f,
                      cudaStream_t stream = 0);

#endif
