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
//! \param u distortion-corrected x
//! coordinates (no boundary check performed!)
//! \param v distortion-corrected y
//! coordinates (no boundary check performed!)
//!

//!
//! \brief Compute distortion correction map
//! from undistorted image coordinates \c u and
//! \c v to original image coordinates \c x and
//! \c y, using same process as OpenCV undistort)
//! \param x original x coordinates (used as
//! initial values for \c u)
//! \param y original y coordinates (used as
//! initial values for \c v)
//! \param cols not used for border checks, so
//! resulting \c u values might be out of
//! bounds!
//! \param rows not used for border checks, so
//! resulting \c v values might be out of
//! bounds!
//! \param camera_matrix fx, fy, cx, and cy as
//! in OpenCV undistort
//! \param distortion_coeffs k1, k2, and k3 as
//! in OpenCV undistort
//! \param u for each new position \c u_i, use
//! \c u[u_i] from original image
//! \param v for each new potision \c v_i, use
//! \c v[v_i] from original image
//!
__global__ void undistort(const float * x, const float * y,
                          const size_t cols, const size_t rows,
                          const float * distortion_coeffs,
                          const float * camera_matrix,
                          float * u, float * v)
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

        u[pos] = x[pos];
        u[pos] -= cx;
        u[pos] /= fx;

        v[pos] = y[pos];
        v[pos] -= cy;
        v[pos] /= fy;

        // radial distortion correction, at this point:
        // u = x', and v = y'
        float r2 = powf(u[pos], 2) + powf(v[pos], 2);
        float kr_poly = 1 + k1 * r2 + k2 * powf(r2,2) + k3 * powf(r2,3);

        u[pos] /= kr_poly;
        u[pos] *= fx;
        u[pos] += cx;

        v[pos] /= kr_poly;
        v[pos] *= fy;
        v[pos] += cy;
    }
}

void cuda_undistort(const float * x, const float * y,
                    const size_t cols, const size_t rows,
                    const float * camera_matrix,
                    const float * distortion_coeffs,
                    float * u, float * v,
                    cudaStream_t stream)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(cols, blocks.x), DivUp(rows, blocks.y));
    undistort<<<grid, blocks, 0, stream>>>(x, y,
                                           cols, rows,
                                           distortion_coeffs,
                                           camera_matrix,
                                           u, v);
    getLastCudaError("Undistort kernel launch failed");
    // TODO add resample!
}
