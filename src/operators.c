#include "cpu/cpu.h"
#include "internal.h"
#include <stdio.h>

Op op_create(Device dev, Optype opty, void *config) {
    char *err = NULL;
    int err_len = SIZE_MAX;
    switch (dev) {
        case DevCpu:
            return op_create_cpu(opty, config);
        default:
            PANIC(UnsupportedDevice);
            return NULL;
    }
}
void op_destroy(Op op) {
    op->drop(op);
}

Kn kn_load(Op op, void *rt_ctx) {
    return op->load(op, rt_ctx);
}
void kn_unload(Kn kn) {
    kn->drop(kn);
}

void *fn_get(Kn kn) {
    return kn->fn;
}
