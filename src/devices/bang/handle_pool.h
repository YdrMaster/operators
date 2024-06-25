#ifndef __BANG_HANDLE_POOL_H__
#define __BANG_HANDLE_POOL_H__

#include "cnnl.h"
#include "cnrt.h"
#include "../pool.h"

const Pool<cnnlHandle_t> &get_cnnl_pool();

template<typename T>
void use_cnnl(cnrtQueue_t queue, T const &f) {
    auto &pool = get_cnnl_pool();
    auto handle = pool.pop();
    if (!handle) {
        cnnlCreate(&(*handle));
    }
    cnnlSetQueue(*handle, (cnrtQueue_t) queue);
    f(*handle);
    pool.push(std::move(*handle));
}

#endif // __BANG_HANDLE_POOL_H__
