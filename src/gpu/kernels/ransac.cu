#include <cuda.h>
#include <cuda_runtime.h>
#include "ransac.h"
#include "helper_cuda.h"
#include "cudamath.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/copy.h>

#include <random>

#include "svd.cu"

#define SQ(x) (x)*(x)

using namespace thrust::placeholders;

struct is_valid_feature
{
    __host__ __device__ bool operator()(const float x)
    {
        return x >= 0;
    }
};

__global__ void establish_correspondences(const float* src_x, const float* src_y, const float* dst_x, const float* dst_y,
                                          float* c_src_x, float* c_src_y, float* c_dst_x, float* c_dst_y,
                                          const int * matches, const int num_pts)
{
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    if (idx >= num_pts) return;
    const int match_index = matches[idx];
    if (match_index != -1) {
        c_src_x[idx] = src_x[idx];
        c_src_y[idx] = src_y[idx];
        c_dst_x[idx] = dst_x[match_index];
        c_dst_y[idx] = dst_y[match_index];
    }
    else {
        c_src_x[idx] = -1;
        c_src_y[idx] = -1;
        c_dst_x[idx] = -1;
        c_dst_y[idx] = -1;
    }
}

void align_points(const float* src_x, const float* src_y, const float* dst_x, const float* dst_y,
                  float* c_src_x, float* c_src_y, float* c_dst_x, float* c_dst_y,
                  const int * matches, const int num_pts, cudaStream_t stream)
{
    dim3 blocks(256);
    dim3 grid(DivUp(num_pts, blocks.x));
    establish_correspondences<<<grid, blocks, 0, stream>>>(src_x, src_y, dst_x, dst_y, c_src_x, c_src_y,
                                                           c_dst_x, c_dst_y, matches, num_pts);

}

__device__ int eval_homography(const float* src_x, const float* src_y, const float* dst_x, const float* dst_y,
                               const int num_pts, const float H[9], const float inlier_threshold)
{
    int inliers = 0;

    for(int i=0; i < num_pts; i++)
    {
        if (src_x[i] >= 0) {
            float x = H[0]*src_x[i] + H[1]*src_y[i] + H[2];
            float y = H[3]*src_x[i] + H[4]*src_y[i] + H[5];
            float z = H[6]*src_x[i] + H[7]*src_y[i] + H[8];
            x /= z;
            y /= z;
            float dist_sq = (dst_x[i] - x) * (dst_x[i] - x) +
                    (dst_y[i] - y) * (dst_y[i] - y);

            if (dist_sq < inlier_threshold)
                inliers++;
        }
    }
    return inliers;
}

__device__ int compute_homography_2(const float2 src[4], const float2 dst[4], float ret_H[9])
{
    // This version normalises the data before processing as recommended in the book Multiple View Geometry
    GPU_Matrix X;
    GPU_Matrix V;
    GPU_Vector S;

    X.rows = 9;
    X.cols = 9;

    V.rows = 9;
    V.cols = 9;

    S.size = 3;

    // Normalise the data
    float2 src_mean, dst_mean;
    float src_var = 0.0f;
    float dst_var = 0.0f;

    src_mean.x = (src[0].x + src[1].x + src[2].x + src[3].x)*0.25f;
    src_mean.y = (src[0].y + src[1].y + src[2].y + src[3].y)*0.25f;

    dst_mean.x = (dst[0].x + dst[1].x + dst[2].x + dst[3].x)*0.25f;
    dst_mean.y = (dst[0].y + dst[1].y + dst[2].y + dst[3].y)*0.25f;

    for(int i=0; i < 4; i++) {
        src_var += SQ(src[i].x - src_mean.x) + SQ(src[i].y - src_mean.y);
        dst_var += SQ(dst[i].x - dst_mean.x) + SQ(dst[i].y - dst_mean.y);
    }

    src_var *= 0.25f;
    dst_var *= 0.25f;

    float src_scale = sqrt(2.0f) / sqrt(src_var);
    float dst_scale = sqrt(2.0f) / sqrt(dst_var);

    for(int i=0; i < 4; i++) {
        float srcx = (src[i].x - src_mean.x)*src_scale;
        float srcy = (src[i].y - src_mean.y)*src_scale;

        float dstx = (dst[i].x - dst_mean.x)*dst_scale;
        float dsty = (dst[i].y - dst_mean.y)*dst_scale;

        int y1 = (i*2 + 0)*9;
        int y2 = (i*2 + 1)*9;

        // First row
        X.data[y1 + 0] = 0.0f;
        X.data[y1 + 1] = 0.0f;
        X.data[y1 + 2] = 0.0f;

        X.data[y1 + 3] = -srcx;
        X.data[y1 + 4] = -srcy;
        X.data[y1 + 5] = -1.0f;

        X.data[y1 + 6] = dsty*srcx;
        X.data[y1 + 7] = dsty*srcy;
        X.data[y1 + 8] = dsty;

        // Second row
        X.data[y2 + 0] = srcx;
        X.data[y2 + 1] = srcy;
        X.data[y2 + 2] = 1.0f;

        X.data[y2 + 3] = 0.0f;
        X.data[y2 + 4] = 0.0f;
        X.data[y2 + 5] = 0.0f;

        X.data[y2 + 6] = -dstx*srcx;
        X.data[y2 + 7] = -dstx*srcy;
        X.data[y2 + 8] = -dstx;
    }

    // Fill the last row
    float srcx = (src[3].x - src_mean.x)*src_scale;
    float srcy = (src[3].y - src_mean.y)*src_scale;
    float dstx = (dst[3].x - dst_mean.x)*dst_scale;
    float dsty = (dst[3].y - dst_mean.y)*dst_scale;

    X.data[8*9 + 0] = -dsty*srcx;
    X.data[8*9 + 1] = -dsty*srcy;
    X.data[8*9 + 2] = -dsty;

    X.data[8*9 + 3] = dstx*srcx;
    X.data[8*9 + 4] = dstx*srcy;
    X.data[8*9 + 5] = dstx;

    X.data[8*9 + 6] = 0.0f;
    X.data[8*9 + 7] = 0.0f;
    X.data[8*9 + 8] = 0.0f;

    bool ret = linalg_SV_decomp_jacobi(&X, &V, &S);

    float H[9];
    float divisor = V.data[8*9 + 8];

    H[0] = V.data[0*9 + 8]/divisor;
    H[1] = V.data[1*9 + 8]/divisor;
    H[2] = V.data[2*9 + 8]/divisor;
    H[3] = V.data[3*9 + 8]/divisor;
    H[4] = V.data[4*9 + 8]/divisor;
    H[5] = V.data[5*9 + 8]/divisor;
    H[6] = V.data[6*9 + 8]/divisor;
    H[7] = V.data[7*9 + 8]/divisor;
    H[8] = 1;

    // Undo the transformation using inv(dst_transform) * H * src_transform
    // Matrix operation expanded out using wxMaxima
    float s1 = src_scale;
    float s2 = dst_scale;

    float tx1 = src_mean.x;
    float ty1 = src_mean.y;

    float tx2 = dst_mean.x;
    float ty2 = dst_mean.y;

    ret_H[0] = s1*tx2*H[6] + s1*H[0]/s2;
    ret_H[1] = s1*tx2*H[7] + s1*H[1]/s2;
    ret_H[2] = tx2*(H[8] - s1*ty1*H[7] - s1*tx1*H[6]) + (H[2] - s1*ty1*H[1] - s1*tx1*H[0])/s2;

    ret_H[3] = s1*ty2*H[6] + s1*H[3]/s2;
    ret_H[4] = s1*ty2*H[7] + s1*H[4]/s2;
    ret_H[5] = ty2*(H[8] - s1*ty1*H[7] - s1*tx1*H[6]) + (H[5] - s1*ty1*H[4] - s1*tx1*H[3])/s2;

    ret_H[6] = s1*H[6];
    ret_H[7] = s1*H[7];
    ret_H[8] = H[8] - s1*ty1*H[7] - s1*tx1*H[6];

    return ret;
}

__device__ int compute_homography(const float2 src[4], const float2 dst[4], float ret_H[9])
{
    // This version normalises the data before processing as recommended in the book Multiple View Geometry
    GPU_Matrix X;
    GPU_Matrix V;
    GPU_Vector S;

    X.rows = 9;
    X.cols = 9;

    V.rows = 9;
    V.cols = 9;

    S.size = 3;

    for(int i=0; i < 4; i++) {
        float srcx = src[i].x;
        float srcy = src[i].y;

        float dstx = dst[i].x;
        float dsty = dst[i].y;

        int y1 = (i*2 + 0)*9;
        int y2 = (i*2 + 1)*9;

        // First row
        X.data[y1 + 0] = 0.0f;
        X.data[y1 + 1] = 0.0f;
        X.data[y1 + 2] = 0.0f;

        X.data[y1 + 3] = -srcx;
        X.data[y1 + 4] = -srcy;
        X.data[y1 + 5] = -1.0f;

        X.data[y1 + 6] = dsty*srcx;
        X.data[y1 + 7] = dsty*srcy;
        X.data[y1 + 8] = dsty;

        // Second row
        X.data[y2 + 0] = srcx;
        X.data[y2 + 1] = srcy;
        X.data[y2 + 2] = 1.0f;

        X.data[y2 + 3] = 0.0f;
        X.data[y2 + 4] = 0.0f;
        X.data[y2 + 5] = 0.0f;

        X.data[y2 + 6] = -dstx*srcx;
        X.data[y2 + 7] = -dstx*srcy;
        X.data[y2 + 8] = -dstx;
    }

    // Fill the last row
    float srcx = src[3].x;
    float srcy = src[3].y;
    float dstx = dst[3].x;
    float dsty = dst[3].y;

    X.data[8*9 + 0] = -dsty*srcx;
    X.data[8*9 + 1] = -dsty*srcy;
    X.data[8*9 + 2] = -dsty;

    X.data[8*9 + 3] = dstx*srcx;
    X.data[8*9 + 4] = dstx*srcy;
    X.data[8*9 + 5] = dstx;

    X.data[8*9 + 6] = 0.0f;
    X.data[8*9 + 7] = 0.0f;
    X.data[8*9 + 8] = 0.0f;

    bool ret = linalg_SV_decomp_jacobi(&X, &V, &S);

    float divisor = V.data[8*9 + 8];

    ret_H[0] = V.data[0*9 + 8]/divisor;
    ret_H[1] = V.data[1*9 + 8]/divisor;
    ret_H[2] = V.data[2*9 + 8]/divisor;
    ret_H[3] = V.data[3*9 + 8]/divisor;
    ret_H[4] = V.data[4*9 + 8]/divisor;
    ret_H[5] = V.data[5*9 + 8]/divisor;
    ret_H[6] = V.data[6*9 + 8]/divisor;
    ret_H[7] = V.data[7*9 + 8]/divisor;
    ret_H[8] = 1;

    /*ret_H[0] = 1;
    ret_H[1] = 0;
    ret_H[2] = V.data[2*9 + 8]/divisor;
    ret_H[3] = 0;
    ret_H[4] = 1;
    ret_H[5] = V.data[5*9 + 8]/divisor;
    ret_H[6] = 0;
    ret_H[7] = 0;
    ret_H[8] = 1;*/

    return ret;
}

__device__ void compute_translation(const float2 src, const float2 dst, float ret_H[9])
{
    ret_H[0] = ret_H[4] = ret_H[8] = 1;
    ret_H[1] = ret_H[3] = ret_H[6] = ret_H[7] = 0;
    ret_H[2] = dst.x - src.x;
    ret_H[5] = dst.y - src.y;
}


__global__ void translation_kernel(const float* src_x, const float* src_y, const float* dst_x, const float* dst_y,
                                   const int src_size, float* homographies, int* inliers_idx,
                                   const int* rand_list, const int iterations, const float inlier_threshold)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= iterations) return;

    int rand_idx[2];
    rand_idx[0] = rand_list[idx * 2];
    rand_idx[1] = rand_list[idx * 2 + 1];

    float2 src, dst;
    src.x = src_x[rand_idx[0]];
    src.y = src_y[rand_idx[0]];

    dst.x = dst_x[rand_idx[1]];
    dst.y = dst_y[rand_idx[1]];

    float* H = &homographies[idx * 9];
    // Now we are ready to compute the homography
    compute_translation(src, dst, H);
    // Now evaluate the estimated homography on all the points
    inliers_idx[idx] = eval_homography(src_x, src_y, dst_x, dst_y, src_size, H, inlier_threshold);
}

__global__ void homography_kernel(const float* src_x, const float* src_y, const float* dst_x, const float* dst_y,
                                  const int src_size, float* homographies, int* inliers_idx,
                                  const int* rand_list, const int iterations, const float inlier_threshold)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= iterations) return;

    int rand_idx[4];
    rand_idx[0] = rand_list[idx * 4];
    rand_idx[1] = rand_list[idx * 4 + 1];
    rand_idx[2] = rand_list[idx * 4 + 2];
    rand_idx[3] = rand_list[idx * 4 + 3];

    // Check for duplicates
    if(rand_idx[0] == rand_idx[1]) return;
    if(rand_idx[0] == rand_idx[2]) return;
    if(rand_idx[0] == rand_idx[3]) return;
    if(rand_idx[1] == rand_idx[2]) return;
    if(rand_idx[1] == rand_idx[3]) return;
    if(rand_idx[2] == rand_idx[3]) return;

    float2 src[4], dst[4];
    for(int i = 0; i < 4; i++) {
        src[i].x = src_x[rand_idx[i]];
        src[i].y = src_y[rand_idx[i]];
        dst[i].x = dst_x[rand_idx[i]];
        dst[i].y = dst_y[rand_idx[i]];
    }

    float* H = &homographies[idx * 9];
    // Now we are ready to compute the homography
    compute_homography(src, dst, H);
    // Now evaluate the estimated homography on all the points
    inliers_idx[idx] = eval_homography(src_x, src_y, dst_x, dst_y, src_size, H, inlier_threshold);
}

bool ransac_translation(float *src_x, float *src_y, float *dst_x, float *dst_y,
                        const int src_size, const int dst_size, float inlier_threshold,
                        int iterations, float *homography, cudaStream_t stream)
{
    // Compute a random list on CPU
    thrust::host_vector<int> cpu_rand_list (iterations * 2);
    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_int_distribution<int> dist(0, src_size-1);
    thrust::device_ptr<float> temp = thrust::device_pointer_cast(src_x);
    thrust::host_vector<float> src_x_data(temp, temp + src_size);
    int v;
    int c = 0;
    do
    {
        v = dist(engine);
        if (src_x_data[v]  >= 0) {
            cpu_rand_list[c] = v;
            ++c;
        }
    } while (c < (iterations * 2));

    // Copy the random list to GPU
    thrust::device_vector<int> gpu_rand_list = cpu_rand_list;

    int * rand_list = thrust::raw_pointer_cast(&gpu_rand_list[0]);
    // Allocate space for the homographies
    thrust::device_vector<float> gpu_homographies(iterations * 9, 0);
    float * homographies = thrust::raw_pointer_cast(&gpu_homographies[0]);

    // Allocate space for the inliers count
    thrust::device_vector<int> gpu_inliers_idx(iterations, 0);
    int * inliers_idx = thrust::raw_pointer_cast(&gpu_inliers_idx[0]);
    const int threads = 256;
    const int blocks = DivUp(iterations, threads);
    translation_kernel<<<blocks, threads, 0, stream>>>(src_x, src_y, dst_x, dst_y, src_size,
                                                       homographies, inliers_idx,
                                                       rand_list, iterations, inlier_threshold);
    getLastCudaError("RANSAC launch failed");

    thrust::device_vector<int>::iterator iter = thrust::max_element(gpu_inliers_idx.begin(),
                                                                    gpu_inliers_idx.end());


    unsigned int position = iter - gpu_inliers_idx.begin();
    float * h = thrust::raw_pointer_cast(&gpu_homographies[position*9]);
    cudaMemcpy(homography, h, 9 * sizeof(float), cudaMemcpyDeviceToDevice);
    return true;
}


bool ransac_homography(float *src_x, float *src_y, float *dst_x, float *dst_y,
                       const int src_size, const int dst_size, float inlier_threshold,
                       int iterations, float *homography, cudaStream_t stream)
{
    // Compute a random list on CPU
    thrust::host_vector<int> cpu_rand_list (iterations * 4);
    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_int_distribution<int> dist(0, src_size-1);
    thrust::device_ptr<float> temp = thrust::device_pointer_cast(src_x);
    thrust::host_vector<float> src_x_data(temp, temp + src_size);
    int v;
    int c = 0;
    do
    {
        v = dist(engine);
        if (src_x_data[v]  >= 0) {
            cpu_rand_list[c] = v;
            ++c;
        }
    } while (c < (iterations * 4));

    // Copy the random list to GPU
    thrust::device_vector<int> gpu_rand_list = cpu_rand_list;

    int * rand_list = thrust::raw_pointer_cast(&gpu_rand_list[0]);
    // Allocate space for the homographies
    thrust::device_vector<float> gpu_homographies(iterations * 9, 0);
    float * homographies = thrust::raw_pointer_cast(&gpu_homographies[0]);

    // Allocate space for the inliers count
    thrust::device_vector<int> gpu_inliers_idx(iterations, 0);
    int * inliers_idx = thrust::raw_pointer_cast(&gpu_inliers_idx[0]);
    const int threads = 256;
    const int blocks = DivUp(iterations, threads);
    homography_kernel<<<blocks, threads, 0, stream>>>(src_x, src_y, dst_x, dst_y, src_size,
                                                      homographies, inliers_idx,
                                                      rand_list, iterations, inlier_threshold);
    getLastCudaError("RANSAC launch failed");

    thrust::device_vector<int>::iterator iter = thrust::max_element(gpu_inliers_idx.begin(),
                                                                    gpu_inliers_idx.end());


    unsigned int position = iter - gpu_inliers_idx.begin();
    float * h = thrust::raw_pointer_cast(&gpu_homographies[position*9]);
    cudaMemcpy(homography, h, 9 * sizeof(float), cudaMemcpyDeviceToDevice);


    /*std::cout << position << " " << gpu_inliers_idx[position] << std::endl;

    position = position * 9;

    std::cout << gpu_homographies[position + 0] << " " << gpu_homographies[position + 1] << " " << gpu_homographies[position + 2] << std::endl;
    std::cout << gpu_homographies[position + 3] << " " << gpu_homographies[position + 4] << " " << gpu_homographies[position + 5] << std::endl;
    std::cout << gpu_homographies[position + 6] << " " << gpu_homographies[position + 7] << " " << gpu_homographies[position + 8] << std::endl;
    std::cout.flush();*/

    //if (*iter < 100) return false;

    return true;


}
