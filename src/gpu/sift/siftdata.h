#ifndef __SIFTPOINT_H__
#define __SIFTPOINT_H__

#include <thrust/device_vector.h>

//!
//! \brief Commonly used value in literature
//!
#define SIFT_VECTOR_SIZE    128

//!
//! \brief Heuristically selected for fitting a homography,
//! but can be increased for images with a richer set of
//! features
#define MAX_DESCRIPTORS     1024

//!
//! \brief Keeps keypoints, descriptors and matches
//!
struct SiftData
{
    //!
    //! \brief SIFT descriptors of keypoints
    //!
    thrust::device_vector<float>    _desc;

    //!
    //! \brief Indices kept for matching keypoints
    //!
    thrust::device_vector<int>      _match_indexes;

    //!
    //! \brief x coordinates of keypoints
    //!
    thrust::device_vector<float>    _x;

    //!
    //! \brief y coordinates of keypoints
    //!
    thrust::device_vector<float>    _y;

    //!
    //! \brief Provide access to the underlying data to pass to
    //! raw kernels
    //! \sa _x
    //!
    float *                         _x_ptr;

    //!
    //! \brief Provide access to the underlying data to pass to
    //! raw kernels
    //! \sa _y
    //!
    float *                         _y_ptr;

    //!
    //! \brief Provide access to the underlying data to pass to
    //! raw kernels
    //! \sa _match_indexes
    //!
    int *                           _match_indexes_ptr;

    //!
    //! \brief Number of keypoints
    //!
    int                             _num_items;

    //!
    //! \brief Maximum number of keypoints supported
    //! \sa initialize_data
    //!
    int                             _capacity;

    //!
    //! \brief Do nothing
    //!
    SiftData() {}

    //!
    //! \brief Initialise with passed \c capacity
    //! \param capacity
    //! \sa initialize_data
    //!
    SiftData(int capacity);

    //!
    //! \brief Clear all data
    //! \sa clear_data
    //!
    ~SiftData();

    //!
    //! \brief Perform a shallow copy from \c in
    //! \param in
    //!
    void copy_from(const SiftData & in);

    //!
    //! \brief Allocate enough memory to accomodate at most
    //! \c capacity keypoints
    //! \param capacity
    //! \sa clear_data
    //!
    void initialize_data(int capacity = MAX_DESCRIPTORS);

    //!
    //! \brief Clear all data, that is deallocate
    //! \sa initialize_data
    //!
    void clear_data();
};

#endif
