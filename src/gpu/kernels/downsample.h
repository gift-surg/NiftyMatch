#ifndef __DOWNSAMPLE_H__
#define __DOWNSAMPLE_H__

#include <cuda.h>
#include <cuda_runtime.h>

//!
//! \brief Downsample an image by a factor of 2 for the scale-space
//! representation.
//! \param result
//! \param result_width provided explicitly to avoid index-out-of-bounds
//! type errors
//! \param  provided explicitly to avoid index-out-of-bounds
//! type errors
//! \param source
//! \param source_width
//! \param source_height
//! \param stream
//! \sa PyramidData
//!
template<typename DataType>
void downsample_by_2(DataType* result, const int result_width, const int result_height,
                     const DataType* source, const int source_width, const int source_height,
                     cudaStream_t stream=0);

#endif
