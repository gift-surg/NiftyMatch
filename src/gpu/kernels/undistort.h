#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

//!
//! \brief Correct image positions in \c x and
//! \c y for distortion (using same process as
//! OpenCV undistort), and save new positions
//! in \c x_d and \c y_d
//! \param x
//! \param y
//! \param cols not used for border checks, so
//! resulting \c x_u values might be out of
//! bounds!
//! \param rows not used for border checks, so
//! resulting \c y_u values might be out of
//! bounds!
//! \param camera_matrix fx, fy, cx, and cy as
//! in OpenCV undistort
//! \param distortion_coeffs k1, k2, and k3 as
//! in OpenCV undistort
//! \param x_u
//! \param y_u
//!
void cuda_undistort(const float * x, const float * y,
                    const size_t cols, const size_t rows,
                    const float * camera_matrix,
                    const float * distortion_coeffs,
                    float * x_u, float * y_u,
                    cudaStream_t stream = 0);
