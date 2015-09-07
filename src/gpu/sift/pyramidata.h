#ifndef __PYRAMID_DATA_H__
#define __PYRAMID_DATA_H__

#include "siftparams.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime_api.h>

#define MAX_KERNEL_LENGTH 91

class PyramidData
{
public:
    PyramidData()
        :_num_levels(0), _num_dogs(0), _num_kernels(0)
    {}

    PyramidData(const SiftParams & params);
    ~PyramidData() {}
    void initialize(const SiftParams & params);
    void clear();
    void gpu_collate_keypoints_for_level(int level, int num_pixels);

public:
    thrust::device_vector<float>        _octave[20];
    thrust::device_vector<float>        _dog[19];
    thrust::device_vector<float4>       _key_pts[19];
    thrust::device_vector<float2>       _orientations[19];
    thrust::device_vector<float>        _base_kernel;
    int                                 _base_radius;
    thrust::device_vector<float>        _kernels[20];
    std::vector<int>                    _kernel_radii;
    thrust::device_vector<float>        _buffer;

    thrust::device_vector<float2>       _grad;
    thrust::device_vector<float4>       _collated_kpts[19];

    int                                 _num_levels;
    int                                 _num_dogs;
    int                                 _num_kernels;

private:
    void generate_kernels(const SiftParams &params);
    void create_kernel_for_sigma(float sigma, thrust::device_vector<float> & result,
                                 int & radius);
};

#endif
