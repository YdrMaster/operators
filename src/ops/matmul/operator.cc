#include "../utils.h"
#include "ops/matmul/matmul.h"

#ifdef ENABLE_CPU
#include "cpu/matmul_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/matmul_cuda.h"
#include <cublas_v2.h>
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/matmul_cnnl.h"
#endif
#ifdef ENABLE_ASCEND_NPU
// #include "ascend/matmul_aclnn.h"
#endif

struct MatmulDescriptor {
    Device device;
};

__C MatmulDescriptor *createMatmulDescriptor(Device device, void *config) {
#ifdef ENABLE_CPU

#endif
#ifdef ENABLE_NV_GPU
    
#endif
#ifdef ENABLE_CAMBRICON_MLU

#endif
        
    return nullptr;
}

__C void destroyMatmulDescriptor(MatmulDescriptor *descriptor) {
  
#ifdef ENABLE_CPU
   
#endif
#ifdef ENABLE_NV_GPU
  
#endif
#ifdef ENABLE_CAMBRICON_MLU
      
#endif
      
}

__C void matmul(MatmulDescriptor *descriptor, Tensor c, float beta, Tensor a,
                Tensor b, float alpha, void *stream) {
#ifdef ENABLE_CPU
   
#endif
#ifdef ENABLE_NV_GPU
   
#endif
#ifdef ENABLE_CAMBRICON_MLU
       
#endif
   
}
