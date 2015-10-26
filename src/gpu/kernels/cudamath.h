#ifndef __CUDA_MATH_H__
#define __CUDA_MATH_H__

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

//!
//! \brief Divide \c a by \c b, using upward
//! rounding. For instance:
//! \code
//! d = DivUp(5, 2); // d is 3
//! \endcode
//! \param a
//! \param b
//! \return \code ceil(a/b) \endcode
//!
extern "C" int DivUp(int a, int b);

//!
//! \brief Divide \c a by \c b, using integer
//! division (downward rounding)
//! \param a
//! \param b
//! \return
//!
extern "C" int DivDown(int a, int b);

//!
//! \brief Align \c a up to \c b
//! \param a
//! \param b
//! \return if \c a not aligned to \c b:
//! \code a + a % b \endcode
//!
extern "C" int AlignUp(int a, int b);

//!
//! \brief Align \c a down to \c b
//! \param a
//! \param b
//! \return if \c a not aligned to \c b:
//! \code a - a % b \endcode
//!
extern "C" int AlignDown(int a, int b);

//!
//! \brief Subtract image \c B from \c A
//! \param A
//! \param B
//! \param C resulting image saved here
//! \param width
//! \param height
//! \param stream
//!
template<typename TYPE>
void subtract(const TYPE * A, const TYPE * B,
              TYPE * C,
              const int width, const int height,
              cudaStream_t stream = 0);

//!
//! \brief Compute gradients in x and y directions
//! within \c source
//! \param source
//! \param result gradients saved here
//! \param width
//! \param height
//! \param stream
//!
template<typename TYPE>
void gradient(const TYPE * source,
              float2 * result,
              const int width, const int height,
              cudaStream_t stream = 0);

//!
//! \brief Floating-point modulus of \c x by \c 2*pi
//! \param x
//! \return
//!
inline __host__ __device__ float mod_2pi_f (float x)
{
    while (x > (float)(2 * M_PI)) x -= (float) (2 * M_PI) ;
    while (x < 0.0F) x += (float) (2 * M_PI);
    return x;
}
#endif
