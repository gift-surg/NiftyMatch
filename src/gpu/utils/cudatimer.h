#ifndef __CUDA_TIMER_H__
#define __CUDA_TIMER_H__

#include "macros.h"
#include <cuda.h>
#include <cuda_runtime.h>

//!
//! \brief Provides functionality for timing CUDA operations.
//!
//! Relies on CUDA event recording.
//! \sa <a href="http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__EVENT_ga324d5ce3fbf46899b15e5e42ff9cfa5.html">cudaEventRecord()</a>
//!
class CudaTimer
{
public:
    //!
    //! \brief CudaTimer
    //! \param stream for recording events
    //!
    CudaTimer(cudaStream_t stream=0);

    //!
    //! \brief Register beginning of an event
    //!
    void start();

    //!
    //! \brief Register end of last event launched
    //! \return time elapsed in \c milliseconds
    //!
    float stop();

private:
    cudaEvent_t         _start;
    cudaEvent_t         _stop;
    cudaStream_t        _stream;

    DISALLOW_COPY_AND_ASSIGNMENT(CudaTimer);
};

#endif
