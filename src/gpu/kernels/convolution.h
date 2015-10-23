#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

#include <cuda.h>
#include <cuda_runtime.h>

//!
//! \brief Convolve \c image with \c kernel
//! \param result convolved image saved here
//! \param image
//! \param buffer for optimal performance, e.g.
//! via exploiting spatial locality
//! \param width
//! \param height
//! \param kernel
//! \param kernel_radius
//! \param stream
//!
template<typename TYPE>
void convolve(TYPE * result, const TYPE * image, TYPE * buffer,
              const int width, const int height,
              const float * kernel, const int kernel_radius,
              cudaStream_t stream = 0);

#endif
