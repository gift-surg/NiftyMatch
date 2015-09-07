#include "siftfunctions.h"
#include "stridedrange.cu"
#include "cudamath.h"
#include "helper_cuda.h"
#include "cudatex2D.h"
#include "match.h"
#include "transpose.h"
#include "keypoint.h"
#include "orientation.h"
#include "descriptor.h"

#include <memory>
#include <thrust/host_vector.h>

void compute_sift_matches(SiftData *A, SiftData *B, float *distance,
                          float ambiguity, cudaStream_t stream)
{
    const int A_size = A->_num_items;
    const int B_size = B->_num_items;

    thrust::device_vector<float> transposed_source (A_size * SIFT_VECTOR_SIZE);
    float * A_data = thrust::raw_pointer_cast(&A->_desc[0]);
    float * B_data = thrust::raw_pointer_cast(&B->_desc[0]);
    float * transposed_A_data = thrust::raw_pointer_cast(&transposed_source[0]);
    transpose<float>(transposed_A_data, A_data, SIFT_VECTOR_SIZE, A_size);
    getLastCudaError("Transpose kernel launch failed");

    thrust::device_vector<float> transposed_distance(A_size * B_size);
    float * transposed_result = thrust::raw_pointer_cast(&transposed_distance[0]);
    compute_brute_force_distance<float>(transposed_A_data, A_size,
                                        B_data, B_size, SIFT_VECTOR_SIZE, transposed_result);

    // Transpose the result back
    transpose<float>(distance, transposed_result, A_size, B_size);
    getLastCudaError("Transpose kernel launch failed");


    int * result = thrust::raw_pointer_cast(&A->_match_indexes[0]);
    get_sift_matches<float>(distance, A_size, B_size, B_size, result, ambiguity, stream);
}

void compute_sift_octave(PyramidData & pydata, SiftParams & params, int octave)
{
}

void compute_dog(PyramidData & pydata, const int octave_width, const int octave_height,
                 cudaStream_t stream)
{
    for (int i = 0; i < pydata._num_dogs; ++i) {
        float * f = thrust::raw_pointer_cast(&(pydata._octave[i][0]));
        float * s = thrust::raw_pointer_cast(&(pydata._octave[i+1][0]));
        float * r = thrust::raw_pointer_cast(&(pydata._dog[i][0]));
        subtract(s, f, r, octave_width, octave_height, stream);
    }
}

void compute_gradients(PyramidData & pydata, const SiftParams & params, const int octave_width,
                       const int octave_height, cudaStream_t stream)
{
    float2* g = thrust::raw_pointer_cast(&pydata._grad[0]);
    const int offset = octave_width * octave_height;

    for (int i = params._level_min + 1; i <= params._level_max - 2; ++i) {
        float * f = thrust::raw_pointer_cast(&(pydata._octave[i+1][0]));
        gradient(f, &g[i*offset], octave_width, octave_height, stream);
    }
}

void compute_keypoints_with_mask(PyramidData &pydata, SiftParams &params, cudaTextureObject_t mask,
                                 const int octave, const int octave_width, const int octave_height,
                                 cudaStream_t stream)
{
    cudaArray ** arrays = new cudaArray*[pydata._num_dogs];
    // Create the corresponding textures
    std::unique_ptr<CudaTex2D[]> textures (new CudaTex2D[pydata._num_dogs]);
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    for (int i = 0; i < pydata._num_dogs; ++i) {
        // Create the array and copy the data
        checkCudaErrors(cudaMallocArray(&arrays[i], &desc, octave_width, octave_height));
        checkCudaErrors(cudaMemcpyToArray(arrays[i], 0, 0, thrust::raw_pointer_cast(&pydata._dog[i][0]),
                        octave_width * octave_height * sizeof(float), cudaMemcpyDeviceToDevice));
        // Bind to texture
        textures[i].set(arrays[i], cudaReadModeElementType);
    }

    float xper = std::pow(2.0, octave);
    for (int i = 1; i < pydata._num_dogs - 1; ++i) {
        thrust::fill(pydata._key_pts[i-1].begin(), pydata._key_pts[i-1].end(),
                make_float4(-1.0f, -1.0f, -1.0f, -1.f));
        float4 * key_pts = thrust::raw_pointer_cast(&pydata._key_pts[i-1][0]);
        find_keypoints(textures[i], mask, textures[i-1], textures[i+1], octave_width, octave_height,
                params._peak_threshold, params._edge_threshold, xper, params._sigma_0,
                params._num_dog_levels, (i-1), key_pts, stream);
    }

    // Delete the arrays
    for (int i = 0; i < pydata._num_dogs; ++i) {
        cudaFreeArray(arrays[i]);
        arrays[i] = 0;
    }
    delete[] arrays;
}

void compute_keypoints(PyramidData & pydata, const SiftParams & params, const int octave, const int octave_width,
                       const int octave_height, cudaStream_t stream)
{
    // Generate the texture s for all the DoG images here.
    // This will be fast for extracting keypoints
    cudaArray ** arrays = new cudaArray*[pydata._num_dogs];
    // Create the corresponding textures
    std::unique_ptr<CudaTex2D[]> textures (new CudaTex2D[pydata._num_dogs]);
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    for (int i = 0; i < pydata._num_dogs; ++i) {
        // Create the array and copy the data
        checkCudaErrors(cudaMallocArray(&arrays[i], &desc, octave_width, octave_height));
        checkCudaErrors(cudaMemcpyToArray(arrays[i], 0, 0, thrust::raw_pointer_cast(&pydata._dog[i][0]),
                        octave_width * octave_height * sizeof(float), cudaMemcpyDeviceToDevice));
        // Bind to texture
        textures[i].set(arrays[i], cudaReadModeElementType);
    }

    float xper = std::pow(2.0, octave);
    for (int i = 1; i < pydata._num_dogs - 1; ++i) {
        thrust::fill(pydata._key_pts[i-1].begin(), pydata._key_pts[i-1].end(),
                make_float4(-1.0f, -1.0f, -1.0f, -1.f));
        float4 * key_pts = thrust::raw_pointer_cast(&pydata._key_pts[i-1][0]);
        find_keypoints(textures[i], textures[i-1], textures[i+1], octave_width, octave_height,
                params._peak_threshold, params._edge_threshold, xper, params._sigma_0,
                params._num_dog_levels, (i-1), key_pts, stream);
    }

    // Delete the arrays
    for (int i = 0; i < pydata._num_dogs; ++i) {
        cudaFreeArray(arrays[i]);
        arrays[i] = 0;
    }
    delete[] arrays;
}

void compute_orientations(PyramidData & pydata, const SiftParams & params,
                          const int octave, const int octave_width,
                          const int octave_height, cudaStream_t stream)
{
    float xper = std::pow(2.0, octave);
    const int num_pixels_for_octave = octave_width * octave_height;
    for (int i = 0; i < params._num_dog_levels; ++i) {
        // This is super slow!
        pydata.gpu_collate_keypoints_for_level(i, num_pixels_for_octave);
        if (pydata._orientations[i].size() == 0) return;
        float4 * key_pts = thrust::raw_pointer_cast(&pydata._collated_kpts[i][0]);
        float2 * orientations = thrust::raw_pointer_cast(&pydata._orientations[i][0]);
        float2 * grad = thrust::raw_pointer_cast(&pydata._grad[0]);
        detect_orientations(key_pts, grad, pydata._orientations[i].size(), octave_width,
                            octave_height, 1.5f, xper, orientations, stream);
    }
}

void compute_descriptors(PyramidData & pydata, const SiftParams & params, const int octave,
                         const int octave_width, const int octave_height,
                         SiftData & data, cudaStream_t stream)
{
    float xper = std::pow(2.0, octave);
    for (int i = 0; i < params._num_dog_levels; ++i) {
        if (pydata._orientations[i].size() == 0) return;
        float4 * key_pts = thrust::raw_pointer_cast(&pydata._collated_kpts[i][0]);
        float2 * orientations = thrust::raw_pointer_cast(&pydata._orientations[i][0]);
        float2 * grad = thrust::raw_pointer_cast(&pydata._grad[0]);

        int num_pts = pydata._orientations[i].size();
        int capacity = data._desc.size() / SIFT_VECTOR_SIZE;
        if (num_pts + data._num_items > capacity) {
            num_pts = capacity - data._num_items;
        }

        if (num_pts > 0) {
            float * desc = thrust::raw_pointer_cast(&data._desc[data._num_items * SIFT_VECTOR_SIZE]);
            float * x = thrust::raw_pointer_cast(&data._x[data._num_items]);
            float * y = thrust::raw_pointer_cast(&data._y[data._num_items]);
            compute_sift_descriptors(key_pts, orientations, grad, num_pts, octave_width,
                                     octave_height, params._num_dog_levels, xper, desc,
                                     x, y, stream);
            data._num_items += num_pts;
        }
    }
}
