#ifndef CPU_HANDLE_H
#define CPU_HANDLE_H

#include "status.h"
#include "operators.h"

struct CpuContext{
    Device device;
};
typedef struct CpuContext* CpuHandle_t;

infiniopStatus_t createCpuHandle(CpuHandle_t* handle_ptr);

#endif
