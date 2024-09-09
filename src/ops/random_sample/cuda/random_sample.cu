#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "random_sample.cuh"
#include <cub/block/block_reduce.cuh>
template<class T, int BLOCK_DIM>
__global__ void random_sample_kernel(int *result,
                                     T const *probs,
                                     float topp,
                                     int topk,
                                     float temperature, int voc) {
    topk = cub::Min()(topk, voc);
    if (blockDim.x >= topk) {

        __shared__ T tmpMax[BLOCK_DIM];
        __shared__ int tmpInd[BLOCK_DIM];
        __shared__ T srcTopk[BLOCK_DIM];
        T data = static_cast<T>(-__FLT_MAX__);
        int dataInd = -1;
        for (int i = threadIdx.x; i < voc; i += blockDim.x) {
            if (data < probs[i]) {
                data = probs[i];
                dataInd = i;
            }
        }
        tmpMax[threadIdx.x] = data;
        tmpInd[threadIdx.x] = dataInd;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int i = 0; i < topk; i++) {
                for (int j = i + 1; j < BLOCK_DIM; j++) {
                    if (tmpMax[i] < tmpMax[j]) {
                        T tmp = tmpMax[i];
                        tmpMax[i] = tmpMax[j];
                        tmpMax[j] = tmp;

                        int indexTmp = tmpInd[i];
                        tmpInd[i] = tmpInd[j];
                        tmpInd[j] = indexTmp;
                    }
                }
            }
        }
        __syncthreads();

        float sum_s = 0.0f;
        for (int i = threadIdx.x; i < voc; i += BLOCK_DIM) {
            sum_s += __expf(static_cast<float>(probs[i] - tmpMax[0]) / temperature);
        }
        __shared__ float sum_inverse_total;

        typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        float block_sum = BlockReduce(temp_storage).Reduce(sum_s, cub::Sum());
        if (threadIdx.x == 0) {
            sum_inverse_total = __fdividef(1.0F, block_sum);//高精度除法
        }

        __syncthreads();
        tmpMax[threadIdx.x] = static_cast<T>(__expf(static_cast<float>(tmpMax[threadIdx.x] - tmpMax[0]) / temperature) * sum_inverse_total);
        if (blockIdx.x == 0) {
            srcTopk[0] = tmpMax[0];
            for (int i = 1; i < topk; i++) {
                srcTopk[i] = srcTopk[i - 1] + tmpMax[i];
            }
        }
        int end = 0;
        for (end = 0; end < topk; end++) {
            if (srcTopk[end] >= static_cast<T>(topp)) {
                break;
            }
        }
        if (end < topk - 1) {
            end += 1;
        } else {
            end = topk;
        }
        T randomVal = 0.75;
        randomVal *= srcTopk[end - 1];
        for (int i = 0; i < end; i++) {
            if (randomVal < srcTopk[i]) {
                result[0] = tmpInd[i];
                break;
            }
        }
    }
}

void random_sample_nv_gpu_f16(RandomSampleCudaDescriptor_t desc, void *workspace, void *result,
                              void *probs,
                              float topp,
                              int topk,
                              float temperature,
                              void *stream) {
    int voc = desc->voc;
    int BLOCK_DIM = 1024;
    int num_blocks = (voc + BLOCK_DIM - 1) / BLOCK_DIM;
    random_sample_kernel<half, 1024><<<num_blocks, BLOCK_DIM, 0, (cudaStream_t) stream>>>((int *) (result),
                                                                                          (half *) (probs),
                                                                                          topp,
                                                                                          topk,
                                                                                          temperature, voc);
}

infiniopStatus_t cudaRandomSample(RandomSampleCudaDescriptor_t desc,
                                  void *workspace,
                                  uint64_t workspace_size,
                                  void *result,
                                  void *probs,
                                  float topp,
                                  int topk,
                                  float temperature,
                                  void *stream) {
    if (cudaSetDevice(desc->device_id) != cudaSuccess) {
        return STATUS_BAD_DEVICE;
    }
    if (dtype_eq(desc->dtype, F16)) {
        random_sample_nv_gpu_f16(desc, workspace, result, probs, topp, topk, temperature, stream);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}