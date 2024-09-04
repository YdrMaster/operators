#include "../utils.h"
#include "ops/rotary_embedding/rotary_embedding.h"

#ifdef ENABLE_CPU
#include "cpu/rotary_embedding_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/rotary_embedding.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/rotary_embedding_cnnl.h"
#endif
#ifdef ENABLE_ASCEND_NPU
// #include "ascend/rotary_embedding.h"
#endif

struct RotaryEmbeddingDescriptor {
    Device device;
};

__C void *createRotaryEmbeddingDescriptor(Device device, void *config) {
  
#ifdef ENABLE_CPU
      
#endif
#ifdef ENABLE_NV_GPU
    
#endif
#ifdef ENABLE_CAMBRICON_MLU
 
#endif
#ifdef ENABLE_ASCEND_NPU

#endif

    return nullptr;
};

__C void destroyRotaryEmbeddingDescriptor(RotaryEmbeddingDescriptor *descriptor) {

#ifdef ENABLE_CPU
      
#endif
#ifdef ENABLE_NV_GPU
    
#endif
#ifdef ENABLE_CAMBRICON_MLU
      
#endif
#ifdef ENABLE_ASCEND_NPU
     
#endif
      
}

__C void rotaryEmbedding(RotaryEmbeddingDescriptor *descriptor, Tensor t, Tensor pos, float theta, void *stream) {
#ifdef ENABLE_CPU
     
#endif
#ifdef ENABLE_NV_GPU
        
#endif
#ifdef ENABLE_CAMBRICON_MLU
     
#endif
#ifdef ENABLE_ASCEND_NPU

#endif

};
