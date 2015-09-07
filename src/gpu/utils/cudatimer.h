#ifndef __CUDA_TIMER_H__
#define __CUDA_TIMER_H__

#include "macros.h"
#include <cuda.h>
#include <cuda_runtime.h>


class CudaTimer
{
public:
    CudaTimer(cudaStream_t stream=0);
    void start();
    float stop();

private:
    cudaEvent_t         _start;
    cudaEvent_t         _stop;
    cudaStream_t        _stream;

    DISALLOW_COPY_AND_ASSIGNMENT(CudaTimer);
};

#endif
