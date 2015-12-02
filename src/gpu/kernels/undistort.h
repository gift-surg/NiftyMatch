#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

//!
//! \brief Compute distortion correction map
//! from undistorted image coordinates \c u and
//! \c v to original image coordinates \c x and
//! \c y, using same process as OpenCV undistort
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
//! \param camera_matrix \c fx, \c fy, \c cx,
//! and \c cy as in OpenCV undistort
//! \param distortion_coeffs \c k1, \c k2, and
//! \c k3 as in OpenCV undistort
//! \param u for each new position \c u_i, use
//! \c u[u_i] from original image
//! \param v for each new position \c v_i, use
//! \c v[v_i] from original image
//! \param stream
//!
void cuda_undistort(const float * x, const float * y,
                    const size_t cols, const size_t rows,
                    const float * camera_matrix,
                    const float * distortion_coeffs,
                    float * u, float * u,
                    cudaStream_t stream = 0);
