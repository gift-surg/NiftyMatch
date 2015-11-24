#pragma once

//!
//! \brief Correct image in \c data for
//! distortion (using same process as OpenCV
//! undistort), and save result in \c data
//! as well
//! \param data
//! \param cols
//! \param rows
//! \param camera_matrix
//! \param distortion_coeffs
//! \param stream
//!
void cuda_undistort(float * data,
                    const size_t cols, const size_t rows,
                    const float * camera_matrix,
                    const float * distortion_coeffs,
                    cudaStream_t stream = 0);
