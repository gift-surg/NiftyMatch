#include "cudatimer.h"

CudaTimer::CudaTimer(cudaStream_t stream)
    :_stream(stream)
{
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
}

void CudaTimer::start()
{
    cudaEventRecord(_start, _stream);
}

float CudaTimer::stop()
{
    cudaEventRecord(_stop, _stream);
    cudaEventSynchronize(_stop);
    float et;
    cudaEventElapsedTime(&et, _start, _stop);
    return et;
}
