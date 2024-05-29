#include "operators.h"
#include <cstdio>

int main(int argc, char **argv) {
    printf("Hello, world!\n");

    auto op = op_create(DevCpu, OpRmsNorm, nullptr);
    printf("op: %p\n", op);

    auto kn = kn_load(op, nullptr);
    printf("kn: %p\n", kn);

    auto fn = reinterpret_cast<RmsNormFn>(fn_get(kn));
    printf("fn: %p\n", fn);

    fn(kn,
       MutTensor{TensorLayout{}, nullptr},
       ConstTensor{TensorLayout{}, nullptr},
       ConstTensor{TensorLayout{}, nullptr},
       1e-4);

    kn_unload(kn);
    op_destroy(op);
    return 0;
}
