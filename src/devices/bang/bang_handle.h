#ifndef BANG_HANDLE_H
#define BANG_HANDLE_H

#include "cnnl.h"
#include "cnrt.h"
#include "status.h"
#include "../pool.h"
#include "device.h"

struct BangContext {
    Device device;
    int device_id;
    Pool<cnnlHandle_t> cnnl_handles;
};
typedef struct BangContext *BangHandle_t;

infiniopStatus_t createBangHandle(BangHandle_t *handle_ptr, int device_id);

template<typename T>
void use_cnnl(BangHandle_t bang_handle, cnrtQueue_t queue, T const &f) {
    auto &pool = bang_handle->cnnl_handles;
    auto handle = pool.pop();
    if (!handle) {
        cnrtSetDevice(bang_handle->device_id);
        cnnlCreate(&(*handle));
    }
    cnnlSetQueue(*handle, (cnrtQueue_t) queue);
    f(*handle);
    pool.push(std::move(*handle));
}

#endif
