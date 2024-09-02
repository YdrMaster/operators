#ifndef ASCEND_HANDLE_H
#define ASCEND_HANDLE_H

#include "../pool.h"
#include "device.h"
#include "status.h"
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>

struct AscendContext {
    Device device;
    int device_id;
    Pool<aclOpExecutor *> aclnn_handles;
};
typedef struct AscendContext *AscendHandle_t;

infiniopStatus_t createAscendHandle(AscendHandle_t *handle_ptr, int device_id);

template<typename T>
void use_aclnn_workspace(AscendHandle_t handle, T const &f) {
    auto &pool = handle->aclnn_handles;
    auto executor = pool.pop();
    f(&(*executor));
    aclSetAclOpExecutorRepeatable(*executor);
    pool.push(std::move(*executor));
}

template <typename T>
void use_aclnn_compute(AscendHandle_t handle, T const &f) {
    auto &pool = handle->aclnn_handles;
    auto executor = pool.pop();
    f(*executor);
    aclDestroyAclOpExecutor(*executor);
    pool.push(std::move(*executor));
}

#endif
