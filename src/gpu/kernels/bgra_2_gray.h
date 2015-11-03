#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

//!
//! \brief Get a grayscale representation of a
//! \c bgra image
//! \param bgra
//! \param output grayscale image saved here
//! \param width
//! \param height
//! \param stream
//!
template<typename OutputType>
void cuda_grayscale(const uchar4 * bgra,
                    OutputType * output,
                    const int width, const int height,
                    cudaStream_t stream = 0);

//!
//! \brief Extract desired \c channel of \c bgra
//! image into \c output
//! \param bgra
//! \param output
//! \param width
//! \param height
//! \param channel
//! \param stream
//!
template<typename OutputType>
void cuda_extract_channel(const uchar4 * bgra,
                          OutputType * output,
                          const int width, const int height,
                          const int channel,
                          cudaStream_t stream = 0);

//!
//! \brief Copy \c input values into desired
//! \c channel of \c bgra
//! \param bgra
//! \param input
//! \param width
//! \param height
//! \param channel
//! \param stream
//!
template<typename InputType>
void cuda_put_channel(uchar4 * bgra,
                      const InputType * input,
                      const int width, const int height,
                      const int channel,
                      cudaStream_t stream = 0);

//!
//! \brief Set opacity of all \c bgra entries to a
//! constant \c val
//! \param bgra
//! \param width
//! \param height
//! \param val
//! \param stream
//!
void cuda_set_alpha_to_const(uchar4 * bgra,
                             const int width, const int height,
                             const unsigned char val = 255,
                             cudaStream_t stream = 0);
