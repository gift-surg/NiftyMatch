#include "cast.h"
#include "cudamath.h"
#include <vector_types.h>
#include <helper_cuda.h>


template<typename FROM, typename TO>
__global__ void cast(const FROM * src,
                     const size_t cols, const size_t rows,
                     TO * dst, TO max_val = 0)
{
    const size_t i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const size_t j = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    const size_t pos = j * cols + i; // cols = width, rows = height

    if (0 <= pos and pos < cols * rows) {
        dst[pos] = (max_val != 0 and src[pos] >= max_val) ?
                    max_val :
                    (TO)(src[pos]);
    }
}

template<typename FROM, typename TO>
void cuda_cast(const FROM * src,
               const size_t cols, const size_t rows,
               TO * dst, TO max_val,
               cudaStream_t stream)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(cols, blocks.x), DivUp(rows, blocks.y));
    cast<<<grid, blocks, 0, stream>>>(src, cols, rows, dst, max_val);
    getLastCudaError("Cast kernel launch failed");
}

template
void cuda_cast<float, unsigned char>(const float * src,
                                     const size_t cols, const size_t rows,
                                     unsigned char * dst, unsigned char max_val,
                                     cudaStream_t stream);
