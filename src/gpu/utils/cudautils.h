#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

//!
//! \brief Provides utility functions for some
//! basic CUDA operations
//!
class CudaUtils
{
public:
    //!
    //! \brief Get ID of best GPU (with maximum
    //! GFLOPS)
    //! \return
    //!
    static int get_max_flops_device_id();

    //!
    //! \brief Set default GPU for use by CUDA
    //! to the one with \c device_id
    //! \param device_id
    //!
    static void setup_CUDA(int device_id);

private:
    static int          _max_gflops_device_id;
};

#endif
