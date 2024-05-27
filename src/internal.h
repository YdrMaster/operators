#ifndef __INTERNAL_H__
#define __INTERNAL_H__

#include "operators.h"

struct Operator {
    Device device;
    Optype optype;
    void *config;

    Kn (*load)(struct Operator *, void *rt_ctx);
    void (*drop)(struct Operator *);
};

struct Kernel {
    Device device;
    Optype optype;
    void *rt_ctx;

    Fn fn;
    void (*drop)(struct Kernel *);
};

#endif// __INTERNAL_H__
