#include "cudatex2D.h"
#include "helper_cuda.h"

void CudaTex2D::set(cudaArray *array, cudaTextureReadMode read_mode)
{
    release();

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeBorder;
    tex_desc.addressMode[1] = cudaAddressModeBorder;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = read_mode;
    tex_desc.normalizedCoords = false;
    checkCudaErrors(cudaCreateTextureObject(&_tex, &res_desc, &tex_desc, NULL));
}

CudaTex2D::CudaTex2D(cudaArray * array)
{
    set(array);
}

void CudaTex2D::release()
{
    if (_tex) {
        cudaDestroyTextureObject(_tex);
        _tex = 0;
    }
}

CudaTex2D::~CudaTex2D()
{
    cudaDestroyTextureObject(_tex);
}
