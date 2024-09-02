#ifndef INFINIOP_HANDLE_H
#define INFINIOP_HANDLE_H

#include "device.h"
#include <cudnn.h>

typedef struct HandleStruct {
    Device device;
    cudnnHandle_t cudnn_handle;
} HandleStruct;

typedef HandleStruct *infiniopHandle_t;

#endif
