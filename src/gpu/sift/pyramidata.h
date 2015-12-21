#ifndef __PYRAMID_DATA_H__
#define __PYRAMID_DATA_H__

#include "siftparams.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime_api.h>

#define MAX_KERNEL_LENGTH 91

//!
//! \brief Keeps the scale-space data used and generated in SIFT.
//!
class PyramidData
{
public:
    //!
    //! \brief Default constructor leaving all initialisation to client
    //!
    PyramidData()
        :_num_octaves(0), _num_dogs(0), _num_kernels(0)
    {}

    //!
    //! \brief Initialise using passed \c params
    //! \param params
    //! \sa initialize
    //!
    PyramidData(const SiftParams & params);

    //!
    //! \brief Does nothing
    //!
    ~PyramidData() {}

    //!
    //! \brief Initialise using passed \c params
    //! \param params
    //!
    void initialize(const SiftParams & params);

    //!
    //! \brief clear all used data
    //!
    void clear();

    //!
    //! \brief Collates keypoints for \c level into a (smaller) memory
    //! for faster processing
    //! \param level
    //! \param num_pixels
    //!
    void gpu_collate_keypoints_for_level(int level, int num_pixels);

public:
    //!
    //! \brief Octaves used in difference of Gaussian and gradient
    //! computations
    //!
    thrust::device_vector<float>        _octave[20];

    //!
    //! \brief Difference of Gaussians
    //!
    thrust::device_vector<float>        _dog[19];

    //!
    //! \brief Keypoints
    //!
    thrust::device_vector<float4>       _key_pts[19];

    //!
    //! \brief Keypoint orientations
    //!
    thrust::device_vector<float2>       _orientations[19];

    //!
    //! \brief For creating kernels
    //! \sa _kernels
    //!
    thrust::device_vector<float>        _base_kernel;

    //!
    //! \brief For creating kernels
    //! \sa _kernels
    //!
    int                                 _base_radius;

    //!
    //! \brief Space for Gaussian kernels created
    //!
    thrust::device_vector<float>        _kernels[20];

    //!
    //! \sa _kernels
    //!
    std::vector<int>                    _kernel_radii;

    thrust::device_vector<float>        _buffer;

    //!
    //! \brief Keypoint gradients
    //!
    thrust::device_vector<float2>       _grad;

    //!
    //! \brief Space for collated keypoints
    //! \sa gpu_collate_keypoints_for_level
    //!
    thrust::device_vector<float4>       _collated_kpts[19];

    //!
    //! \brief Number of scale-space levels
    //!
    int                                 _num_octaves;

    //!
    //! \brief Number of difference of Gaussians
    //!
    int                                 _num_dogs;

    //!
    //! \sa SiftParams._sigmas
    //!
    int                                 _num_kernels;

private:
    void generate_kernels(const SiftParams &params);
    void create_kernel_for_sigma(float sigma, thrust::device_vector<float> & result,
                                 int & radius);
};

#endif
