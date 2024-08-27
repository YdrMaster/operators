#include "tensor_desc.h"
#include "common_ascend.h"
#include <iostream>

/// @brief Create aclnnTensorDescriptor
/// @param aclnnTensorDesc_t* Alias of aclnnTensorDescriptor**
void aclnnCreateTensorDescriptor(aclnnTensorDesc_t *desc) {
    *desc = new aclnnTensorDesc;
    (*desc)->ndim = 0;
    (*desc)->shape = nullptr;
    (*desc)->stride = nullptr;
    (*desc)->offset = 0;
    (*desc)->dataType = aclDataType::ACL_FLOAT;
    (*desc)->format = aclFormat::ACL_FORMAT_ND;
    (*desc)->storageNdim = 0;
    (*desc)->storageShape = nullptr;
}

/// @brief Set value of TensorDescriptor. Call behand aclnnCreateTensorDescriptor
/// @param desc Alias of aclnnTensorDescriptor*.
/// @param shape View shape of Tensor.
/// @param stride View stride of Tensor.
/// @param ndim Dimension of shape and stride.
/// @param offset The offset of the first element of the tensor relative to storage.
/// @param dataType Tensor element type.
/// @param format Tensor data layout.
void aclnnSetTensorDescriptor(aclnnTensorDesc_t desc, int64_t *shape, int64_t *stride, int64_t ndim,
                              int64_t offset, aclDataType dataType, aclFormat format) {
    desc->ndim = ndim;
    desc->shape = shape;
    desc->offset = offset;
    desc->dataType = dataType;
    desc->format = format;
    // Set stride
    if (stride) {
        desc->stride = stride;
    } else {
        auto stride_v = new int64_t(ndim);
        stride_v[ndim - 1] = 1;
        for (int64_t i = ndim - 2; i >= 0; i--) {
            stride_v[i] = shape[i + 1] * stride_v[i + 1];
        }
        desc->stride = stride_v;
    }
    desc->storageNdim = ndim;
    desc->storageShape = shape;
    return;
}

/// @brief Set value of TensorDescriptor acoording to framework's tensorlayout
/// @param desc Alias of aclnnTensorDescriptor*.
/// @param layout Layout of Tensor, see src/tensor.h
void aclnnSetTensorDescriptorFromTensorLayout(aclnnTensorDesc_t desc,
                                              const TensorLayout *layout) {
    // Cast shape's unint64_t to int64_t
    auto dims = new int64_t(layout->ndim);
    for (uint64_t i = 0; i < layout->ndim; i++) {
        dims[i] = static_cast<int64_t>(layout->shape[i]);
    }
    // Cast bytes stride to element stride
    auto strides = new int64_t(layout->ndim);
    for (uint64_t i = 0; i < layout->ndim; i++) {
        strides[i] = layout->strides[i] / (layout->dt).size;
    }
    // TODO: support other element type
    aclDataType dtype = aclDataType::ACL_FLOAT16;
    // TODO: support other tensor format
    aclFormat format = aclFormat::ACL_FORMAT_ND;
    // Set aclnnTensorDescriptor
    aclnnSetTensorDescriptor(desc, dims, strides, layout->ndim,
                             0, dtype, format);
    // char *descStr = aclnnTensorDescToString(desc);
    // if (descStr) {
    //     printf("%s", descStr);
    // } else {
    //     printf("Failed to print.\n");
    // }
}

/// @brief Destory TensorDescriptor
/// @param desc Alias of aclnnTensorDescriptor*.
void aclnnDestoryTensorDescriptor(aclnnTensorDesc_t desc) {
    if (desc) {
        // // Free shape space
        // if (desc->shape) {
        //     delete desc->shape;
        //     desc->shape = nullptr;
        // }
        // // Free stride space
        // if (desc->stride) {
        //     delete desc->stride;
        //     desc->stride = nullptr;
        // }
        // // Free storageShape space
        // if (desc->storageShape) {
        //     delete desc->storageShape;
        //     desc->storageShape = nullptr;
        // }
        // Free aclnnTensorDescriptor
        delete desc;
        desc = nullptr;
    }
}


/// @brief Wrapper of aclCreateTensor. Create aclTensor.
/// See https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha001/apiref/appdevgapi/aclcppdevg_03_0168.html
/// @param desc Alias of aclnnTensorDescriptor*.
/// @param data Data ptr on device global mem.
/// @param tensor Pointer of pointer of aclTensor.
/// @return
int aclnnCreateTensor(aclnnTensorDesc_t desc, void *data, aclTensor **tensor) {
    // char *descStr = aclnnTensorDescToString(desc);
    // if (descStr) {
    //     printf("%s", descStr);
    // } else {
    //     printf("Failed to print.\n");
    // }
    *tensor = aclCreateTensor(desc->shape, desc->ndim, desc->dataType, desc->stride, desc->offset,
                              desc->format, desc->storageShape, desc->storageNdim, data);
    return 0;
}

/// @brief TensorDescriptor's string info
/// @param desc Alias of aclnnTensorDescriptor*.
/// @return String of aclnnTensorDescriptor.
char *aclnnTensorDescToString(const aclnnTensorDesc_t desc) {
    if (!desc) return NULL;

    // Assume bufferSize
    size_t bufferSize = 1024 + desc->ndim * 40 + desc->storageNdim * 40;
    char *buffer = (char *) malloc(bufferSize);
    if (!buffer) return NULL;

    // Write info into buffer
    char *ptr = buffer;
    ptr += sprintf(ptr, "ndim: %" PRId64 "\n", desc->ndim);

    ptr += sprintf(ptr, "shape: [");
    for (int64_t i = 0; i < desc->ndim; ++i) {
        ptr += sprintf(ptr, "%" PRId64, desc->shape[i]);
        if (i < desc->ndim - 1) {
            ptr += sprintf(ptr, ", ");
        }
    }
    ptr += sprintf(ptr, "]\n");

    ptr += sprintf(ptr, "stride: [");
    for (int64_t i = 0; i < desc->ndim; ++i) {
        ptr += sprintf(ptr, "%" PRId64, desc->stride[i]);
        if (i < desc->ndim - 1) {
            ptr += sprintf(ptr, ", ");
        }
    }
    ptr += sprintf(ptr, "]\n");

    ptr += sprintf(ptr, "offset: %" PRId64 "\n", desc->offset);
    ptr += sprintf(ptr, "dataType: %s\n", dataTypeToString(desc->dataType));
    ptr += sprintf(ptr, "format: %s\n", formatToString(desc->format));

    ptr += sprintf(ptr, "storageShape: [");
    for (int64_t i = 0; i < desc->storageNdim; ++i) {
        ptr += sprintf(ptr, "%" PRId64, desc->storageShape[i]);
        if (i < desc->storageNdim - 1) {
            ptr += sprintf(ptr, ", ");
        }
    }
    ptr += sprintf(ptr, "]\n");

    ptr += sprintf(ptr, "storageNdim: %" PRId64 "\n", desc->storageNdim);

    return buffer;
}
