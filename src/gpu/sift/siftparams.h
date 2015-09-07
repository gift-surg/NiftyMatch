#ifndef __SIFT_PARAMS_H__
#define __SIFT_PARAMS_H__

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>

#define     MINIMUM_OCTAVE_SIZE 32

class SiftParams
{
public:
    SiftParams()
        :_width(0), _height(0)
    {}

    SiftParams(int width, int height)
        :_width(width), _height(height), _num_dog_levels(3),
         _sigma_n(0.5f), _peak_threshold(0), _edge_threshold(10.f)
    {
        _level_max = _num_dog_levels + 1;
        _level_min = -1;
        _num_octaves = (int)std::floor(std::log(std::min(width, height) * 2.0/MINIMUM_OCTAVE_SIZE)/std::log(2.0));
        if (_num_octaves <= 0) _num_octaves = 1;

        _sigma_k = std::pow(2.0f, 1.0f/_num_dog_levels);
        _sigma_0 = 1.6f * _sigma_k;
        _sigma_d_0 = _sigma_0 * std::sqrt (1.0 - 1.0 / (_sigma_k * _sigma_k));

        float sa = _sigma_0 * std::pow(_sigma_k, _level_min) ;
        float sb = _sigma_n;

        // This is the base image smoothing factor
        if (sa > sb) _base_smooth = std::sqrt (sa*sa - sb*sb);

        // Smoothing factor for each level
        for (int i = _level_min + 1; i <= _level_max; ++i) _sigmas.push_back(_sigma_d_0 * std::pow(_sigma_k, i));
    }

    int                     _width;
    int                     _height;
    int                     _num_octaves;
    int                     _num_dog_levels;
    int                     _level_max;
    int                     _level_min;
    float                   _sigma_d_0;
    float                   _sigma_k;
    float                   _sigma_0;
    float                   _sigma_n;
    float                   _base_smooth;

    std::vector<float>      _sigmas;

    float                   _peak_threshold;
    float                   _edge_threshold;
};

#endif
