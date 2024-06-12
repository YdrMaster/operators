#include "ops/rotary_embedding/rotary_embedding.h"
#include "tensor.h"
#include <stdio.h>

void test_rms_norm() {
    RotaryEmbeddingDescriptor *descriptor = createRotaryEmbeddingDescriptor(DevNvGpu, NULL);
    rotaryEmbedding(descriptor, (MutTensor){.layout = {}, .data = NULL}, (ConstTensor){.layout = {}, .data = NULL}, 10000.0, NULL);
    destroyRotaryEmbeddingDescriptor(descriptor);
}

int main(int argc, char **argv) {
    test_rms_norm();
    return 0;
}
