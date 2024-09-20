#ifndef __CPU_ROTARY_EMBEDDING_H__
#define __CPU_ROTARY_EMBEDDING_H__

#include "operators.h"
#include "../../../devices/cpu/cpu_handle.h"

struct RoPECpuDescriptor;

typedef struct RoPECpuDescriptor *RoPECpuDescriptor_t;

infiniopStatus_t cpuCreateRoPEDescriptor(CpuHandle_t handle,
                                         RoPECpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t t,
                                         infiniopTensorDescriptor_t pos_ids,
                                         infiniopTensorDescriptor_t sin_table,
                                         infiniopTensorDescriptor_t cos_table);

infiniopStatus_t cpuGetRoPEWorkspaceSize(RoPECpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuRoPE(RoPECpuDescriptor_t desc,
                         void *workspace,
                         uint64_t workspace_size,
                         void *t,
                         void const *pos_ids,
                         void const *sin_table,
                         void const *cos_table,
                         void *stream);

infiniopStatus_t cpuDestroyRoPEDescriptor(RoPECpuDescriptor_t desc);


#endif// __CPU_RMS_NORM_H__
