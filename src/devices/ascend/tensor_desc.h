#ifndef __ACLNN_TENSOR_DESC__
#define __ACLNN_TENSOR_DESC__

#include "../../tensor.h"
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "aclnn/acl_meta.h"
#include <inttypes.h>
#include <vector>

// Aclnn tensor descriptor, 
// used to build aclTensor
struct aclnnTensorDesc {
    int64_t ndim;
    int64_t *shape;
    int64_t *stride;
    int64_t offset;
    aclDataType dataType;
    aclFormat format;
    int64_t *storageShape;
    int64_t storageNdim;
};


typedef aclnnTensorDesc *aclnnTensorDesc_t;

void aclnnCreateTensorDescriptor(aclnnTensorDesc_t *desc);

void aclnnSetTensorDescriptor(aclnnTensorDesc_t desc, int64_t *shape, int64_t *stride, int64_t ndim,
                              int64_t offset, aclDataType dataType, aclFormat format);

void aclnnDestoryTensorDescriptor(aclnnTensorDesc_t desc);

int aclnnCreateTensor(aclnnTensorDesc_t desc, void *data, aclTensor **tensor);

char *aclnnTensorDescToString(const aclnnTensorDesc_t desc);

void aclnnSetTensorDescriptorFromTensorLayout(aclnnTensorDesc_t desc, const TensorLayout *layout);

#endif