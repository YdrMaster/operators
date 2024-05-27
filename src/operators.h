#ifndef __OPERATORS_H__
#define __OPERATORS_H__

#include "data_type.h"
#include "device.h"
#include "optype.h"

#define PANIC(EXPR)                                             \
    printf("Error at %s:%d - %s\n", __FILE__, __LINE__, #EXPR); \
    exit(1)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DeviceEnum Device;
typedef enum OptypeEnum Optype;
typedef struct DataLayout DT;
typedef struct Operator *Op;
typedef struct Kernel *Kn;
typedef void *Fn;

Op op_create(Device, Optype, void *config);
void op_destroy(Op);

Kn kn_load(Op, void *rt_ctx);
void kn_unload(Kn);

void *fn_get(Kn);

#ifdef __cplusplus
}
#endif

#endif// __OPERATORS_H__
