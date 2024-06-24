#ifndef __ASCEND_COMMON_H__
#define __ASCEND_COMMON_H__

#include "acl/acl.h"
#include <cstdio>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)



#ifdef __cplusplus
}
#endif

int64_t *castToInt64_t(uint64_t *in, uint64_t cnt);

int64_t shapeProd(int64_t *in, uint64_t cnt);


#endif
