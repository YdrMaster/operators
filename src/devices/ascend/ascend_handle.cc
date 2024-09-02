#include "ascend_handle.h"

infiniopStatus_t createAscendHandle(AscendHandle_t *handle_ptr, int device_id) {
    uint32_t device_count;
    aclrtGetDeviceCount(&device_count);
    if (device_id >= static_cast<int>(device_count)) {
        return STATUS_BAD_DEVICE;
    }
    auto pool = Pool<aclOpExecutor *>();
    aclOpExecutor *executor = nullptr;
    pool.push(std::move(executor));

    *handle_ptr = new AscendContext{DevAscendNpu, device_id, std::move(pool)};

    return STATUS_SUCCESS;
}
