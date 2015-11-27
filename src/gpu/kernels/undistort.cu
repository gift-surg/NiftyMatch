#include "undistort.h"
#include "cudamath.h"
#include <vector_types.h>
#include <helper_cuda.h>

//!
//! \brief Apply distortion correction as in
//! OpenCV undistort
//! \param x original x coordinates
//! \param y original y coordinates
//! \param length number of coordinates (i.e.
//! length of \c x and \c y)
//! \param k1 see OpenCV undistort
//! \param k2 see OpenCV undistort
//! \param k3 see OpenCV undistort
//! \param fx focal length (x) of camera
//! matrix
//! \param fy focal length (y) of camera
//! matrix
//! \param cx principal point (x) of camera
//! matrix
//! \param cy principal point (y) of camera
//! matrix
//! \param x_u distortion-corrected x
//! coordinates
//! \param y_u distortion-corrected y
//! coordinates
//!
__global__ void undistort(const float * x, const float * y,
                          size_t length,
                          const float k1, const float k2, const float k3,
                          const float fx, const float fy,
                          const float cx, const float cy,
                          float * x_u, float * y_u)
{
    float r = 0, tmp = 0;
    for (size_t i = 0; i < length; i++) {
        r = powf(x[i], 2) + powf(y[i], 2);
        // common coefficient for multiplication
        tmp = 1 + k1 * powf(r, 2) + k2 * powf(r, 4) + k3 * powf(r, 6);

        x_u[i] = (x[i] - cx) / fx;
        x_u[i] *= tmp;
        y_u[i] = (y[i] - cy) / fy;
        y_u[i] *= tmp;
    }
}

void cuda_undistort(const float * x, const float * y,
                    const size_t cols, const size_t rows,
                    const float * camera_matrix,
                    const float * distortion_coeffs,
                    float * x_u, float * y_u,
                    cudaStream_t stream)
{
    // TODO blocks
    dim3 blocks(16, 16);
    // TODO grid
    dim3 grid(DivUp(cols, blocks.x), DivUp(rows, blocks.y));
    float k1 = distortion_coeffs[0],
          k2 = distortion_coeffs[1],
          k3 = distortion_coeffs[2];
    float fx = camera_matrix[0],
          fy = camera_matrix[1],
          cx = camera_matrix[2],
          cy = camera_matrix[3];
    undistort<<<grid, blocks, 0, stream>>>(x, y,
                                           rows * cols,
                                           k1, k2, k3,
                                           fx, fy,
                                           cx, cy,
                                           x_u, y_u);
    getLastCudaError("Undistort kernel launch failed");
    // TODO add resample!
}
