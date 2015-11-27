#include "undistort.h"
#include "cudamath.h"
#include <vector_types.h>
#include <helper_cuda.h>

//!
//! \brief Apply distortion correction as in
//! OpenCV undistort
//! \param x original x coordinates
//! \param y original y coordinates
//! \param cols
//! \param rows
//! \param distortion coeffs k1, k2, k3 as in
//! OpenCV undistort
//! \param camera_matrix fx, fy, cx, cy as in
//! OpenCV undistort
//! \param x_u distortion-corrected x
//! coordinates (no boundary check performed!)
//! \param y_u distortion-corrected y
//! coordinates (no boundary check performed!)
//!
__global__ void undistort(const float * x, const float * y,
                          const size_t cols, const size_t rows,
                          const float * distortion_coeffs,
                          const float * camera_matrix,
                          float * x_u, float * y_u)
{
    const size_t i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const size_t j = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    const size_t pos = j * cols + i; // cols = width, rows = height

    const float k1 = distortion_coeffs[0],
                k2 = distortion_coeffs[1],
                k3 = distortion_coeffs[3];
    const float fx = camera_matrix[0],
                fy = camera_matrix[1],
                cx = camera_matrix[2],
                cy = camera_matrix[3];
    if (0 <= pos and pos < cols * rows) {
        float r2 = powf(x[pos] - cols / 2.0f, 2) + powf(y[pos] - rows / 2.0f, 2);
        // common coefficient for multiplication
        float tmp = 1 + k1 * r2 + k2 * powf(r2, 2) + k3 * powf(r2, 3);
        x_u[pos] = (x[pos] - cx) / fx;
//        x_u[pos] = x[pos];
        x_u[pos] *= tmp;
        x_u[pos] *= fx;
        x_u[pos] += cx;

        y_u[pos] = (y[pos] - cy) / fy;
//        y_u[pos] = y[pos];
        y_u[pos] *= tmp;
        y_u[pos] *= fy;
        y_u[pos] += cy;
    }
}

void cuda_undistort(const float * x, const float * y,
                    const size_t cols, const size_t rows,
                    const float * camera_matrix,
                    const float * distortion_coeffs,
                    float * x_u, float * y_u,
                    cudaStream_t stream)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(cols, blocks.x), DivUp(rows, blocks.y));
    undistort<<<grid, blocks, 0, stream>>>(x, y,
                                           cols, rows,
                                           distortion_coeffs,
                                           camera_matrix,
                                           x_u, y_u);
    getLastCudaError("Undistort kernel launch failed");
    // TODO add resample!
}
