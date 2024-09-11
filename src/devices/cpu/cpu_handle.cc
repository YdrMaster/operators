#include "device.h"
#include "cpu_handle.h"

infiniopStatus_t createCpuHandle(CpuHandle_t* handle_ptr){
    *handle_ptr = new CpuContext{DevCpu};
    return STATUS_SUCCESS;
}
