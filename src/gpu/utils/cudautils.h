#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

class CudaUtils
{
public:
    static int get_max_flops_device_id();
    static void setup_CUDA(int device_id);

private:
    static int          _max_gflops_device_id;
};

#endif
