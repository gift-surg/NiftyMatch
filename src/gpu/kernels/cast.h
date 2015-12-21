#pragma once
#include <cuda_runtime.h>

//!
//! \brief Cast all elements of \c src \c FROM
//! old type to \c TO new type, and save result
//! in \c dst
//! \param src
//! \param cols
//! \param rows
//! \param dst
//! \param max_val overflow value, i.e. if any
//! value of \c src greater than this value, then
//! use this value
//! \param stream
//!
template<typename FROM, typename TO>
void cuda_cast(const FROM * src,
               const size_t cols, const size_t rows,
               TO * dst,
               TO max_val = 0,
               cudaStream_t stream = 0);
