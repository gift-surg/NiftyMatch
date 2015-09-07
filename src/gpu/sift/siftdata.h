#ifndef __SIFTPOINT_H__
#define __SIFTPOINT_H__

#include <thrust/device_vector.h>

#define SIFT_VECTOR_SIZE    128
#define MAX_DESCRIPTORS     1024

struct SiftData
{
    thrust::device_vector<float>    _desc;
    thrust::device_vector<int>      _match_indexes;
    thrust::device_vector<float>    _x;
    thrust::device_vector<float>    _y;

    // Get access to the underlying data to pass to raw kernels
    float *                         _x_ptr;
    float *                         _y_ptr;
    int *                           _match_indexes_ptr;


    int                             _num_items;
    int                             _capacity;

    SiftData() {}
    SiftData(int capacity);
    ~SiftData();
    void copy_from(const SiftData & in);
    void initialize_data(int capacity = MAX_DESCRIPTORS);
    void clear_data();
};

#endif
