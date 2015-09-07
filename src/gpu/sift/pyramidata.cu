#include "pyramidata.h"
#include "exception.h"
#include "helper_math.h"
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <algorithm>
#include <functional>

struct is_valid_kpt
{
    __host__ __device__ bool operator()(const float4 x)
    {
        return x.w >= 0;
    }
};


PyramidData::PyramidData(const SiftParams & params)
    :_num_levels(0), _num_dogs(0), _num_kernels(0)
{
    initialize(params);
}

void PyramidData::initialize(const SiftParams & params)
{
    clear();

    _num_levels = params._level_max - params._level_min + 1;
    if (_num_levels > 20) RUNTIME_EXCEPTION("Maximum bumber of levels is 20.");
    const int num_pixels = params._width * params._height;

    for (int i = 0; i < _num_levels; ++i) {
        _octave[i] = thrust::device_vector<float>(num_pixels);
    }
    _num_dogs = params._level_max - params._level_min;

    for (int i = 0; i < _num_dogs; ++i) {
        _dog[i] = thrust::device_vector<float>(num_pixels);
    }

    for (int i = 0; i < params._num_dog_levels; ++i) {
        _key_pts[i] = thrust::device_vector<float4>(num_pixels, make_float4(-1, -1, -1, -1));
        _collated_kpts[i] = thrust::device_vector<float4>(num_pixels, make_float4(-1, -1, -1, -1));
    }

    _grad = thrust::device_vector<float2>(num_pixels * _num_dogs, make_float2(0, 0));

    _buffer = thrust::device_vector<float>(num_pixels);
    generate_kernels(params);
}


void PyramidData::clear()
{
    for (int i = 0; i < _num_levels; ++i) {
        _octave[i].clear();
    }
    _grad.clear();

    for (int i = 0; i < _num_dogs; ++i) {
        _dog[i].clear();
    }

    for (int i = 0; i < _num_kernels; ++i) {
        _kernels[i].clear();
    }

    for (int i = 0; i < _num_dogs - 2; ++i) {
        _key_pts[i].clear();
        _orientations[i].clear();
    }

    if (!_buffer.empty()) {
        _buffer.clear();
    }

    _kernel_radii.clear();
    _num_levels = _num_dogs = _num_kernels = 0;
}




void PyramidData::gpu_collate_keypoints_for_level(int level, int num_pixels)
{
    size_t new_size = thrust::copy_if(_key_pts[level].begin(), _key_pts[level].begin() + num_pixels,
                                      _collated_kpts[level].begin(), is_valid_kpt()) -
                                      _collated_kpts[level].begin();

    _orientations[level] = thrust::device_vector<float2>(new_size, make_float2(-1, -1));
}


void PyramidData::generate_kernels(const SiftParams & params)
{
    create_kernel_for_sigma(params._base_smooth, _base_kernel, _base_radius);
    _num_kernels = params._sigmas.size();
    int rad;
    for (int i = 0; i < _num_kernels; ++i) {
        create_kernel_for_sigma(params._sigmas[i], _kernels[i], rad);
        _kernel_radii.push_back(rad);
    }
}

void PyramidData::create_kernel_for_sigma(float sigma, thrust::device_vector<float> &result,
                                          int & radius)
{
    const int kernel_radius = (int)(std::ceil(sigma * 4));
    const int kernel_length = 2 * kernel_radius + 1;
    thrust::host_vector<float> kernel;
    kernel.reserve(kernel_length);
    float kernel_sum = 0.f;
    for (int j = 0; j < kernel_length; ++j) {
        float val = ((float)j - kernel_radius)/sigma;
        val = (float) exp(- 0.5 * (val*val));
        kernel.push_back(val);
        kernel_sum += val;
    }
    std::transform(kernel.begin(), kernel.end(), kernel.begin(),
                   std::bind2nd(std::divides<float>(), kernel_sum));
    result = thrust::device_vector<float>(kernel);
    radius = kernel_radius;
}

