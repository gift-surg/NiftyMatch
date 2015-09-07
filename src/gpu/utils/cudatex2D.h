#ifndef __CUDA_TEX2D_H__
#define __CUDA_TEX2D_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include "macros.h"


class CudaTex2D
{
public:
    CudaTex2D()
        :_tex(0)
    {}
    void set(cudaArray * array, cudaTextureReadMode read_mode=cudaReadModeNormalizedFloat);
    void release();
    CudaTex2D(cudaArray * array);
    ~CudaTex2D();

    operator cudaTextureObject_t () const { return _tex; }

private:
    cudaTextureObject_t         _tex;
    DISALLOW_COPY_AND_ASSIGNMENT(CudaTex2D);
};

#endif
