#include "keypoint.h"
#include "cudamath.h"
#include "helper_math.h"
#include "helper_cuda.h"

inline __device__ bool fgt(float a, float b)
{
    return a > b;
}

inline __device__ bool flt(float a, float b)
{
    return a < b;
}

typedef bool(*Op)(float, float);

template<Op op>
__device__ bool is_maxima(float ax, float ay, cudaTextureObject_t current,
                          cudaTextureObject_t down, cudaTextureObject_t up)
{
    float cv = tex2D<float>(current, ax, ay);
    // Current slice
    float pv = tex2D<float>(current, ax - 1.f, ay);

    if (!op(cv, pv)) return false;

    pv = tex2D<float>(current, ax + 1.f, ay);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(current, ax - 1.f, ay - 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(current, ax, ay - 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(current, ax + 1.f, ay - 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(current, ax - 1.f, ay + 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(current, ax, ay + 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(current, ax + 1.f, ay + 1.f);
    if (!op(cv, pv)) return false;

    // Down slice
    pv = tex2D<float>(down, ax, ay);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(down, ax - 1.f, ay);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(down, ax + 1.f, ay);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(down, ax - 1.f, ay - 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(down, ax, ay - 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(down, ax + 1.f, ay - 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(down, ax - 1.f, ay + 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(down, ax, ay + 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(down, ax + 1.f, ay + 1.f);
    if (!op(cv, pv)) return false;

    // Up slice
    pv = tex2D<float>(up, ax, ay);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(up, ax - 1.f, ay);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(up, ax + 1.f, ay);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(up, ax - 1.f, ay - 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(up, ax, ay - 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(up, ax + 1.f, ay - 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(up, ax - 1.f, ay + 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(up, ax, ay + 1.f);
    if (!op(cv, pv)) return false;

    pv = tex2D<float>(up, ax + 1.f, ay + 1.f);
    if (!op(cv, pv)) return false;

    return true;
}

__device__ void subpixel_refinement(const int x, const int y,
                                    cudaTextureObject_t current, cudaTextureObject_t down, cudaTextureObject_t up,
                                    const int width, const int height,
                                    const float peak_threshold, const float edge_threshold, const float xper,
                                    const float sigma_0, const int num_dogs, const int level, float4 * result)
{
    const float ax = x + 0.5;
    const float ay = y + 0.5;
    float c = tex2D<float>(current, ax, ay);

    // Compute gradient
    float fx = 0.5 * (tex2D<float>(current, ax + 1.0, ay) - tex2D<float>(current, ax - 1.0, ay));
    float fy = 0.5 * (tex2D<float>(current, ax, ay + 1.0) - tex2D<float>(current, ax, ay - 1.0));
    float fs = 0.5 * (tex2D<float>(up, ax, ay) - tex2D<float>(down, ax, ay));

    // Compute hessian
    float fxx = tex2D<float>(current, ax + 1.0, ay) + tex2D<float>(current, ax - 1.0, ay) - 2.0 * c;
    float fyy = tex2D<float>(current, ax, ay + 1.0) + tex2D<float>(current, ax, ay - 1.0) - 2.0 * c;
    float fss = tex2D<float>(up, ax, ay) + tex2D<float>(down, ax, ay) - 2.0 * c;

    float fxy = 0.25 * (tex2D<float>(current, ax + 1.0, ay + 1.0) + tex2D<float>(current, ax - 1.0, ay - 1.0) -
                        tex2D<float>(current, ax - 1.0, ay + 1.0) - tex2D<float>(current, ax + 1.0, ay - 1.0));

    float fxs = 0.25 * (tex2D<float>(up, ax + 1.0, ay) + tex2D<float>(down, ax - 1.0, ay) -
                        tex2D<float>(up, ax - 1.0, ay) - tex2D<float>(down, ax + 1.0, ay));

    float fys = 0.25 * (tex2D<float>(up, ax, ay + 1.0) + tex2D<float>(down, ax, ay - 1.0) -
                        tex2D<float>(up, ax, ay - 1.0) - tex2D<float>(down, ax, ay + 1.0));

    float4 A0 = fxx > 0? make_float4(fxx, fxy, fxs, -fx) : make_float4(-fxx, -fxy, -fxs, fx);
    float4 A1 = fxy > 0? make_float4(fxy, fyy, fys, -fy) : make_float4(-fxy, -fyy, -fys, fy);
    float4 A2 = fxs > 0? make_float4(fxs, fys, fss, -fs) : make_float4(-fxs, -fys, -fss, fs);
    float4 temp;

    float max_a = fmaxf(fmaxf(A0.x, A1.x), A2.x);
    if(max_a >= 1e-10) {
        if (max_a == A1.x) {
            temp = A1; A1 = A0; A0 = temp;
        }
        else if (max_a == A2.x) {
            temp = A2; A2 = A0; A0 = temp;
        }
        A0.y /= A0.x;  A0.z /= A0.x;	A0.w/= A0.x;
        A1.y -= A1.x * A0.y;	A1.z -= A1.x * A0.z;	A1.w -= A1.x * A0.w;
        A2.y -= A2.x * A0.y;	A2.z -= A2.x * A0.z;	A2.w -= A2.x * A0.w;

        if(fabs(A2.y) > fabs(A1.y)) {
            temp = A2;	A2 = A1; A1 = temp;
        }

        if(fabs(A1.y) >= 1e-10) {
            A1.z /= A1.y;	A1.w /= A1.y;
            A2.z -= A2.y * A1.z;	A2.w -= A2.y * A1.w;
            if(fabs(A2.z) >= 1e-10) {
                float ds = A2.w / A2.z;
                float dy = A1.w - ds * A1.z;
                float dx = A0.w - ds * A0.z - dy * A0.y;
                float v = c + 0.5 * (dx * fx + dy * fy + ds * fs);
                float s = (fxx+fyy)*(fxx+fyy) / (fxx*fyy - fxy*fxy);

                if ((fabs(v) > peak_threshold) &&
                     s < ((edge_threshold+1)*(edge_threshold+1)/edge_threshold) &&
                    fabs(dx) < 1 && fabs(dy) < 1 && fabs(ds) < 1)
                {
                    result[y * width + x].x = (x + dx) * xper;
                    result[y * width + x].y = (y + dy) * xper;
                    result[y * width + x].z = sigma_0 * pow(2.0, (double)(level + ds)/num_dogs) * xper;
                    result[y * width + x].w = level;
                }
            }
        }
    }
}

// Version without mask
__global__ void detect_keypoints(cudaTextureObject_t current, cudaTextureObject_t down, cudaTextureObject_t up,
                                 const int width, const int height, const float peak_threshold,
                                 const float edge_threshold, const float xper, const float sigma_0,
                                 const int num_dogs, const int level, float4 *result)
{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (x < 1 || x > width - 2 || y < 1 || y > height - 2) return;

    float c = tex2D<float>(current, x + 0.5f, y + 0.5f);

    if ((c <= 0.8f*peak_threshold && is_maxima<flt>(x + 0.5f, y + 0.5f, current, down, up)) ||
        (c >= 0.8f*peak_threshold && is_maxima<fgt>(x + 0.5f, y + 0.5f, current, down, up)))
    {
        subpixel_refinement(x, y, current, down, up, width, height, peak_threshold, edge_threshold,
                            xper, sigma_0, num_dogs, level, result);
    }
}

// Version with mask
__global__ void detect_keypoints(cudaTextureObject_t current, cudaTextureObject_t mask, cudaTextureObject_t down,
                                 cudaTextureObject_t up, const int width, const int height, const float peak_threshold,
                                 const float edge_threshold, const float xper, const float sigma_0,
                                 const int num_dogs, const int level, float4 *result)
{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (x < 1 || x > width - 2 || y < 1 || y > height - 2) return;

    if (tex2D<float> (mask, (x + 0.5f) * xper, (y + 0.5f) * xper) < 1.f) return;

    float c = tex2D<float>(current, x + 0.5f, y + 0.5f);

    if ((c <= 0.8f*peak_threshold && is_maxima<flt>(x + 0.5f, y + 0.5f, current, down, up)) ||
        (c >= 0.8f*peak_threshold && is_maxima<fgt>(x + 0.5f, y + 0.5f, current, down, up)))
    {
        subpixel_refinement(x, y, current, down, up, width, height, peak_threshold, edge_threshold,
                            xper, sigma_0, num_dogs, level, result);
    }
}

void find_keypoints(cudaTextureObject_t current, cudaTextureObject_t mask, cudaTextureObject_t down,
                    cudaTextureObject_t up, const int width, const int height, const float peak_threshold,
                    const float edge_threshold, const float xper, const float sigma_0,
                    const int num_dogs, const int dog, float4 *result, cudaStream_t stream)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(width, blocks.x), DivUp(height, blocks.y));
    detect_keypoints<<<grid, blocks, 0, stream>>>(current, mask, down, up, width, height,
                                                  peak_threshold, edge_threshold, xper,
                                                  sigma_0, num_dogs, dog, result);
    getLastCudaError("Keypoint detection launch failed");
}


void find_keypoints(cudaTextureObject_t current, cudaTextureObject_t down, cudaTextureObject_t up,
                    const int width, const int height, const float peak_threshold,
                    const float edge_threshold, const float xper, const float sigma_0,
                    const int num_dogs, int dog, float4 *result, cudaStream_t stream)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(width, blocks.x), DivUp(height, blocks.y));
    detect_keypoints<<<grid, blocks, 0, stream>>>(current, down, up, width, height,
                                                  peak_threshold, edge_threshold, xper,
                                                  sigma_0, num_dogs, dog, result);
    getLastCudaError("Keypoint detection launch failed");
}
