#ifndef __CUDA_TEX2D_H__
#define __CUDA_TEX2D_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include "macros.h"

//!
//! \brief Facilitates use of CUDA textures for faster computing.
//!
class CudaTex2D
{
public:
    //!
    //! \brief Default constructor with no CUDA texture set up
    //!
    CudaTex2D()
        :_tex(0)
    {}

    //!
    //! \brief Create a CUDA texture by texturing from \c array
    //! \param array
    //! \param read_mode
    //!
    void set(cudaArray * array, cudaTextureReadMode read_mode=cudaReadModeNormalizedFloat);

    //!
    //! \brief Release used CUDA texture memory
    //!
    void release();

    //!
    //! \brief Create a CUDA texture from \c array
    //! \param array
    //! \sa set
    //!
    CudaTex2D(cudaArray * array);

    //!
    //! \brief Destroy any used CUDA texture
    //!
    ~CudaTex2D();

    //!
    //! \brief Return a handle to current CUDA texture
    //!
    operator cudaTextureObject_t () const { return _tex; }

private:
    cudaTextureObject_t         _tex;
    DISALLOW_COPY_AND_ASSIGNMENT(CudaTex2D);
};

#endif
