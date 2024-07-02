#include "swiglu_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../devices/bang/handle_pool.h" 
#include "../../utils.h"
#include "cnrt.h"

SwigluBangDescriptor::SwigluBangDescriptor(Device device) {
    this->device = device;
    get_cnnl_pool();
}

void swiglu_cnnl_f16(SwigluBangDescriptor *descriptor, Tensor gate, Tensor up, void *stream) {
    ASSERT_EQ(gate.layout->ndim, 2);
    ASSERT_EQ(up.layout->ndim, 2);
    ASSERT_EQ(gate.layout->shape[0], up.layout->shape[0]);
    ASSERT_EQ(gate.layout->shape[1], up.layout->shape[1]);

    setCnnlTensor(descriptor->gateDesc, gate.layout);

    std::vector<int> dims(gate.layout->ndim);
    size_t inputSizeInBytes = 1;
    for (uint64_t i = 0; i < gate.layout->ndim; i++) {
        dims[i] = static_cast<int>(gate.layout->shape[i]);
        inputSizeInBytes *= dims[i];
    }
    dims[gate.layout->ndim - 1] *= 2;
    inputSizeInBytes *= (2 * sizeof(uint16_t));
    cnnlSetTensorDescriptor(descriptor->inDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                            dims.size(), dims.data());

    void *input;
    cnrtMalloc(&input, inputSizeInBytes);

    cnnlSetActivationDescriptor_v6(descriptor->actDesc, CNNL_ACTIVATION_SILU,
                                   CNNL_ACTIVATION_HIGH_PRECISION,
                                   CNNL_NOT_PROPAGATE_NAN,
                                   0.0, 0, 0.0, 0.0, true, true);
    cnnlSetBiasActivationGluDescriptor(descriptor->opDesc, descriptor->actDesc,
                                       CNNL_BIAS_ACTIVATION_GLU_ALGO_V2);

    void *concatWorkspace;
    
    use_cnnl((cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 size_t concatWorkspaceSize;
                 cnnlGetConcatWorkspaceSize(handle, 2, &concatWorkspaceSize);
                 cnrtMalloc(&concatWorkspace, concatWorkspaceSize);

                 cnnlTensorDescriptor_t inputsDesc[2] = {descriptor->gateDesc, descriptor->gateDesc};
                 const void *const inputsData[2] = {gate.data, up.data};
                 cnnlConcat(handle, 2, -1, inputsDesc, inputsData,
                            concatWorkspace, concatWorkspaceSize, descriptor->inDesc, input);

                 cnnlBiasActivationGluForward_v2(handle, descriptor->opDesc, descriptor->inDesc, input,
                                                 nullptr, nullptr, descriptor->gateDesc, gate.data);
             });

    cnrtFree(concatWorkspace);
    cnrtFree(input);
}
