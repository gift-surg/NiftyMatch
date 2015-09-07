#include "cuda_runtime.h"
#include "cuda.h"

int main()
{
    int device_count = 0;
    cudaError_t cuda_result_code = cudaGetDeviceCount(&device_count);

    // Error when running cudaGetDeviceCount
    if(cuda_result_code != cudaSuccess) // cudaSuccess=0
        return EXIT_FAILURE;

    // Returns an error if no cuda card has been detected
    if(device_count == 0)
        return EXIT_FAILURE;
    
    int gpu_device_count = 0;    
    struct cudaDeviceProp properties;
    for (int device = 0; device < device_count; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999) /* 9999 means emulation only */
            ++gpu_device_count;
    }

    if (gpu_device_count > 0) return EXIT_SUCCESS;
    else return EXIT_FAILURE;
}
