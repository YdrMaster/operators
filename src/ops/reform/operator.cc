#include "../utils.h"
#include "ops/reform/reform.h"

#ifdef ENABLE_CPU
#include "cpu/reform_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/reform.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/reform_bang.h"
#endif
#ifdef ENABLE_ASCEND_NPU
// #include "ascend/reform.h"
#endif

struct ReformDescriptor {
    Device device;
};

__C ReformDescriptor *createReformDescriptor(Device device, void *config) {
#ifdef ENABLE_CPU
#endif

#ifdef ENABLE_NV_GPU
#endif

#ifdef ENABLE_CAMBRICON_MLU

#endif
#ifdef ENABLE_ASCEND_NPU
    
#endif
       
    return nullptr;
}

__C void destroyReformDescriptor(ReformDescriptor *descriptor) {
#ifdef ENABLE_CPU
      
#endif
#ifdef ENABLE_NV_GPU
       
#endif
#ifdef ENABLE_CAMBRICON_MLU
       
#endif
#ifdef ENABLE_ASCEND_NPU

#endif
   
}

__C void reform(ReformDescriptor *descriptor, Tensor y, Tensor x, void *stream) {
#ifdef ENABLE_CPU

#endif
#ifdef ENABLE_NV_GPU

#endif
#ifdef ENABLE_CAMBRICON_MLU

#endif
#ifdef ENABLE_ASCEND_NPU

#endif

};
