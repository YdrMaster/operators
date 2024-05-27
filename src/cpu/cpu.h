#ifndef __CPU_H__
#define __CPU_H__

#include "../internal.h"

#ifdef __cplusplus
extern "C" {
#endif

Op op_create_cpu(Optype, void *config);

#ifdef __cplusplus
}
#endif

#endif// __CPU_H__
