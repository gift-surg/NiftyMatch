#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

template<typename OutputType>
void cuda_grayscale(const uchar4* bgra, OutputType * output, const int width, const int height,
                    cudaStream_t stream=0);

template<typename OutputType>
void cuda_extract_channel(const uchar4* bgra, OutputType * output, const int width, const int height,
                          const int channel, cudaStream_t stream=0);


template<typename InputType>
void cuda_put_channel(uchar4* bgra, const InputType * input, const int width, const int height,
                      const int channel, cudaStream_t stream=0);

void cuda_set_alpha_to_const(uchar4* bgra, const int width, const int height,
                             const unsigned char val=255, cudaStream_t stream=0);
