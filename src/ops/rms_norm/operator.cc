#include "../utils.h"
#include "ops/rms_norm/rms_norm.h"

#ifdef ENABLE_CPU
#include "cpu/rms_norm_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/rms_norm.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/rms_norm_bang.h"
#include "bang/rms_norm_cnnl.h"
#endif
#ifdef ENABLE_ASCEND_NPU
// #include "ascend/rms_norm_aclnn.h"
#endif

struct RMSNormDescriptor {
    Device device;
};

__C void *createRMSNormDescriptor(Device device, void *config) {
#ifdef ENABLE_CPU
   
#endif
#ifdef ENABLE_NV_GPU
   
#endif
#ifdef ENABLE_CAMBRICON_MLU
      
#endif
    
    return nullptr;
}

__C void destroyRMSNormDescriptor(RMSNormDescriptor *descriptor) {
#ifdef ENABLE_CPU

#endif
#ifdef ENABLE_NV_GPU
    
#endif
#ifdef ENABLE_CAMBRICON_MLU

#endif

}

__C void rmsNorm(RMSNormDescriptor *descriptor, Tensor y, Tensor x, Tensor w,
                 float epsilon, void *stream) {
#ifdef ENABLE_CPU
  
#endif
#ifdef ENABLE_NV_GPU
 
#endif
#ifdef ENABLE_CAMBRICON_MLU
   
#endif
     
}
