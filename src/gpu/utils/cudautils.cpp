#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cudautils.h"
#include "exception.h"
#include "helper_cuda.h"

int CudaUtils::_max_gflops_device_id = -1;

int CudaUtils::get_max_flops_device_id()
{
    if (_max_gflops_device_id != -1) return _max_gflops_device_id;
    else {
        _max_gflops_device_id = gpuGetMaxGflopsDeviceId();
        return _max_gflops_device_id;
    }
}

void CudaUtils::setup_CUDA(int device_id)
{
    if (cudaSetDevice(device_id) != cudaSuccess) {
        RUNTIME_EXCEPTION("Could not set the CUDA device");
    }

    if (cudaGLSetGLDevice(device_id) != cudaSuccess) {
        RUNTIME_EXCEPTION("Could not set the CUDA OpenGL setup");
    }
}
