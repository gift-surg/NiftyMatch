#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

//!
//! \brief Transpose given \c idata
//! \param odata transposed data is saved here
//! \param idata
//! \param width
//! \param height
//! \param stream
//!
template<typename TYPE>
void transpose(TYPE * odata,
               const TYPE * idata,
               int width, int height,
               cudaStream_t stream = 0);

#endif
