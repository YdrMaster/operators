#include "random_sample_bang.h"
#include "../../utils.h"

infiniopStatus_t bangCreateRandomSampleDescriptor(BangHandle_t handle,
                                                  RandomSampleBangDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result,
                                                  infiniopTensorDescriptor_t probs) {
    if (probs->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!dtype_eq(probs->dt, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!dtype_eq(result->dt, U64))
        return STATUS_BAD_TENSOR_DTYPE;
    int voc = probs->shape[0];
    int rLength = result->shape[0];
    *desc_ptr = new RandomSampleBangDescriptor{
        handle->device,
        handle->device_id,
        probs->dt,
        voc,
        result->dt,
        rLength};

    return STATUS_SUCCESS;
}

infiniopStatus_t bangGetRandomSampleWorkspaceSize(RandomSampleBangDescriptor_t desc, unsigned long int *size) {
    *size = desc->voc * (sizeof(uint64_t) + sizeof(desc->dtype)) + sizeof(desc->dtype);
    return STATUS_SUCCESS;
}

infiniopStatus_t bangDestroyRandomSampleDescriptor(RandomSampleBangDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
