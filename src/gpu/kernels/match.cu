#include <float.h>
#include <thrust/device_vector.h>
#include "helper_cuda.h"
#include "match.h"
#include "cudamath.h"

#if (__CUDA_ARCH__ >= 500) // Maxwell or newer
    #define CHUNK 16
#elif (__CUDA_ARCH__ < 500) // Kepler or older
    #define CHUNK 4
#endif

template<typename TYPE>
__global__ void brute_force_distance(const TYPE* __restrict__ A, const int size_A,
                                     const TYPE* __restrict__ B, const int size_B,
                                     const int vector_dim,
                                     TYPE * D)
{
    extern __shared__ TYPE shared_B[];
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    TYPE result[CHUNK];

    if (bx < gridDim.x - 1) {
        // Move the data into shared memory
        for (int i = 0; i < CHUNK; i++) {
            shared_B[(i*vector_dim)+threadIdx.x] = B[(((bx*CHUNK)+i)*vector_dim)+tx];
        }
        __syncthreads();

        while (tx < size_A)
        {
            for (int i = 0; i < CHUNK; i++) result[i] = 0.0f;

            for (int i = 0; i < vector_dim; i++) {
                TYPE Atemp = A[(size_A*i)+tx]; // Reading elements of Matrix A for L1/L2 cache storage
                for (int j = 0; j < CHUNK; j++) {
                    TYPE temp = Atemp - shared_B[i + (j * vector_dim)];
                    result[j] += temp * temp;
                }
            }

            for (int i = 0; i < CHUNK; i++)
            {
                D[((i+(bx*CHUNK))*size_A)+ tx] = result[i];
            }
            tx += blockDim.x;
        }
    }

    // Special handling for last block
    // Something is wrong here!!!!!!!!!
    else {
        const int new_chunk = size_B - (blockIdx.x * CHUNK);
        for (int i = 0; i < new_chunk; i++) {
            shared_B[(i*vector_dim)+threadIdx.x] = B[(((bx*CHUNK)+i)*vector_dim)+tx];
        }
        __syncthreads();

        while (tx < size_A)
        {
            for (int i = 0; i < new_chunk; i++) result[i] = 0.0f;

            for (int i = 0; i < vector_dim; i++) {
                TYPE Atemp = A[(size_A*i)+tx]; // Reading elements of Matrix A for L1/L2 cache storage
                for (int j = 0; j < new_chunk; j++) {
                    TYPE temp = Atemp - shared_B[i + (j * vector_dim)];
                    result[j] += temp * temp;
                }
            }

            for (int i = 0; i < new_chunk; i++)
            {
                D[((i+(bx*CHUNK))*size_A)+ tx] = result[i];
            }
            tx += blockDim.x;
        }
    }
}

template<typename TYPE>
__global__ void set_matches(TYPE * __restrict__ result, const float * __restrict__ distance,
                            const int num_rows, const int num_cols, const int buffer_width, float ambiguity)
{
    const int i = blockIdx.x * __mul24(blockDim.x, blockDim.y) + __mul24(blockDim.x, threadIdx.y) + threadIdx.x;

    if (i < num_rows) {
        const int row_start = i * buffer_width;
        float min_1_distance = distance[row_start];
        float min_2_distance = 0x7f800000; // infinity
        int min_1 = 0;

        for (int j = 1; j < num_cols; ++j) {
            float current = distance[row_start + j];

            if (current < min_1_distance) {
                min_2_distance = min_1_distance;
                min_1 = j;
                min_1_distance = current;
            }
            else if (current < min_2_distance) {
                min_2_distance = current;
            }
        }

        if (min_2_distance > 0) {
            float a = min_1_distance/min_2_distance;
            if (a < ambiguity) {
                result[i] = min_1;
            }
            else {
                result[i] = -1;
            }
        }
    }
}

template<typename TYPE>
void compute_brute_force_distance(const TYPE* A, const int size_A,
                                  const TYPE* B, const int size_B,
                                  const int sift_vector_size, TYPE* result,
                                  cudaStream_t stream)
{
    checkCudaErrors(cudaFuncSetCacheConfig(brute_force_distance<float>,
                                           cudaFuncCachePreferL1));

    dim3 blocks(DivUp(size_B, CHUNK));
    dim3 threads(sift_vector_size);
    int smemBytes = CHUNK * sift_vector_size * sizeof(TYPE);
    brute_force_distance<TYPE> <<<blocks, threads, smemBytes, stream>>> (A, size_A,
                                                                         B, size_B, sift_vector_size,
                                                                         result);
    getLastCudaError("Brute force distance computation launch failed");
}

template void compute_brute_force_distance<float> (const float*, const int, const float* B, const int,
                                                   const int, float* result, cudaStream_t);

template<typename TYPE>
void get_sift_matches(const TYPE * distance, const int rows, const int cols, const int buffer_width,
                      int *result, float ambiguity, cudaStream_t stream)
{
    checkCudaErrors(cudaFuncSetCacheConfig(set_matches<TYPE>,
                                           cudaFuncCachePreferShared));
    dim3 blocks(16, 16);
    dim3 grid(DivUp(rows, 256));
    set_matches <<<grid, blocks, 0, stream>>> (result, distance, rows, cols, buffer_width, ambiguity);
    getLastCudaError("Set matches launch failed");
}

template void get_sift_matches<float>(const float *, const int, const int, const int,
                                      int *, float, cudaStream_t);
