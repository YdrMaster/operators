#ifndef __DEVICE_H__
#define __DEVICE_H__

enum DeviceEnum {
    DevCpu,
    DevNvGpu,
    DevCambriconMlu,
};

typedef enum DeviceEnum Device;

#endif// __DEVICE_H__
