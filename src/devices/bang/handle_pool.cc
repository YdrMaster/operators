#include <mutex>
#include <vector>
#include "handle_pool.h"

// @deprecated
const Pool<cnnlHandle_t> &get_cnnl_pool() {
    int device_id;
    cnrtGetDevice(&device_id);
    static std::once_flag flag;
    static std::vector<Pool<cnnlHandle_t>> cnnl_pool;
    std::call_once(flag, [&]() {
        unsigned int device_count;
        cnrtGetDeviceCount(&device_count);
        for (auto i = 0; i < static_cast<int>(device_count); i++) {
            auto pool = Pool<cnnlHandle_t>();
            cnnlHandle_t handle;
            cnnlCreate(&handle);
            pool.push(std::move(handle));
            cnnl_pool.emplace_back(std::move(pool));
        }
    });
    return cnnl_pool[device_id];
}
