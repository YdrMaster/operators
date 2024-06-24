#include "ops/rotary_embedding/rotary_embedding.h"
#include "tensor.h"
#include <stdio.h>

void test_rms_norm() {
    void *descriptor = createRotaryEmbeddingDescriptor(DevNvGpu, NULL);
    struct TensorLayout l;
    Tensor t = {l, NULL};
    Tensor t2 = {l, NULL};
    rotaryEmbedding(descriptor, t, t2, 10000.0, NULL);
    destroyRotaryEmbeddingDescriptor(descriptor);
}

int main(int argc, char **argv) {
    test_rms_norm();
    return 0;
}
