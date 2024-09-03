#include "bang_handle.h"

infiniopStatus_t createBangHandle(BangHandle_t *handle_ptr, int device_id) {
    unsigned int device_count;
    cnrtGetDeviceCount(&device_count);
    if (device_id >= static_cast<int>(device_count)) {
        return STATUS_BAD_DEVICE;
    }

    auto pool = std::make_shared<Pool<cnnlHandle_t>>();
    if (cnrtSetDevice(device_id) != cnrtSuccess){
        return STATUS_BAD_DEVICE;
    }
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    pool->push(std::move(handle));

    *handle_ptr = new BangContext{DevCambriconMlu, device_id, std::move(pool)};

    return STATUS_SUCCESS;
}
