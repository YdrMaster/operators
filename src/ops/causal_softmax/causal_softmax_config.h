#ifndef CAUSAL_SOFTMAX_CONFIG_H
#define CAUSAL_SOFTMAX_CONFIG_H

typedef struct CausalSoftmaxCudaConfig {
    // The upper bound of softmax dimension (axis)
    unsigned int max_dim;
} CausalSoftmaxCudaConfig;

#endif // CAUSAL_SOFTMAX_CONFIG_H
