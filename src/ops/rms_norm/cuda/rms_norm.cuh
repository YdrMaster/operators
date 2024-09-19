#ifndef __NV_GPU_RMS_NORM_H__
#define __NV_GPU_RMS_NORM_H__

#include "operators.h"
#include "../../../devices/cuda/cuda_handle.h"

struct RMSNormCudaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    unsigned long int n;
    unsigned long int d;
    unsigned long int stride_y;
    unsigned long int stride_x;
    int8_t w_datatype;
};

typedef struct RMSNormCudaDescriptor *RMSNormCudaDescriptor_t;

infiniopStatus_t cudaCreateRMSNormDescriptor(CudaHandle_t handle,
                                                    RMSNormCudaDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t y_desc,
                                                    infiniopTensorDescriptor_t x_desc,
                                                    infiniopTensorDescriptor_t w_desc,
                                                    int8_t w_datatype);

infiniopStatus_t cudaGetRMSNormWorkspaceSize(RMSNormCudaDescriptor_t desc, unsigned long int *size);

infiniopStatus_t cudaRMSNorm(RMSNormCudaDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *y, void *x, void *w, float epsilon,
                                   void *stream);

infiniopStatus_t cudaDestroyRMSNormDescriptor(RMSNormCudaDescriptor_t desc);

void rms_norm_nv_gpu_f16(RMSNormCudaDescriptor_t desc, void *y, void *x, void *w, float epsilon, void *stream);

#endif// __NV_GPU_RMS_NORM_H__
