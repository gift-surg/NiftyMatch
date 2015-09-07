#include "siftdata.h"

SiftData::SiftData(int capacity)
{
    if (capacity <= 0)
        throw std::runtime_error("Invalid initialization of SIFT data");
    initialize_data(capacity);
}

void SiftData::copy_from(const SiftData & in)
{
    _desc = in._desc;
    _x = in._x;
    _x_ptr = thrust::raw_pointer_cast(&_x[0]);

    _y = in._y;
    _y_ptr = thrust::raw_pointer_cast(&_y[0]);

    _match_indexes = in._match_indexes;
    _match_indexes_ptr = thrust::raw_pointer_cast(&_match_indexes[0]);

    _num_items = in._num_items;
}

SiftData::~SiftData()
{
    clear_data();
}

void SiftData::initialize_data(int capacity)
{
    clear_data();
    _desc = thrust::device_vector<float>(SIFT_VECTOR_SIZE * capacity, 0);
    _match_indexes = thrust::device_vector<int>(capacity, -1);
    _match_indexes_ptr = thrust::raw_pointer_cast(&_match_indexes[0]);

    _x = thrust::device_vector<float>(capacity);
    _x_ptr = thrust::raw_pointer_cast(&_x[0]);

    _y = thrust::device_vector<float>(capacity);
    _y_ptr = thrust::raw_pointer_cast(&_y[0]);

    _capacity = capacity;
    _num_items = 0;
}

void SiftData::clear_data()
{
    _desc.clear();
    _match_indexes.clear();
    _match_indexes_ptr = NULL;
    _x.clear(); _y.clear();
    _x_ptr = _y_ptr = NULL;

    _num_items = _capacity = 0;
}
