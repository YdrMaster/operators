#include "swiglu.h"
#include "../../utils.h"
#include "cnnl.h"
#include "cnnl_extra.h"
#include "cnrt.h"

void swiglu_cambricon_mlu_f16(MutTensor gate, ConstTensor up, void *stream) {
    ASSERT_EQ(gate.layout.ndim, 2);
    ASSERT_EQ(up.layout.ndim, 2);
    ASSERT_EQ(gate.layout.shape[0], up.layout.shape[0]);
    ASSERT_EQ(gate.layout.shape[1], up.layout.shape[1]);

    cnnlTensorDescriptor_t gateDesc, inDesc;
    cnnlCreateTensorDescriptor(&gateDesc);
    cnnlCreateTensorDescriptor(&inDesc);
    setCnnlTensor(gateDesc, gate.layout);

    std::vector<int> dims(gate.layout.ndim);
    size_t inputSizeInBytes = 0;
    for (uint64_t i = 0; i < gate.layout.ndim; i++) {
        dims[i] = static_cast<int>(gate.layout.shape[i]);
        inputSizeInBytes += dims[i] * sizeof(uint16_t);
    }
    dims[gate.layout.ndim - 1] *= 2;
    inputSizeInBytes *= 2;
    cnnlSetTensorDescriptor(inDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                            dims.size(), dims.data());

    void *input;
    cnrtMalloc(&input, inputSizeInBytes);

    auto [handle, queue] = getCnnlHandle(stream);

    size_t concatWorkspaceSize;
    cnnlGetConcatWorkspaceSize(handle, 2, &concatWorkspaceSize);

    void *concatWorkspace;
    cnrtMalloc(&concatWorkspace, concatWorkspaceSize);

    cnnlTensorDescriptor_t inputs[2] = {gateDesc, gateDesc};
    const void *const inputsData[2] = {gate.data, up.data};
    cnnlConcat(handle, 2, -1, inputs, inputsData,
               concatWorkspace, concatWorkspaceSize, inDesc, input);

    cnnlActivationDescriptor_t actDesc;
    cnnlCreateActivationDescriptor(&actDesc);
    cnnlSetActivationDescriptor_v6(actDesc, CNNL_ACTIVATION_SILU,
                                   CNNL_ACTIVATION_HIGH_PRECISION,
                                   CNNL_NOT_PROPAGATE_NAN,
                                   0.0, 0, 0.0, 0.0, true, true);

    cnnlBiasActivationGluDescriptor_t opDesc;
    cnnlCreateBiasActivationGluDescriptor(&opDesc);
    cnnlSetBiasActivationGluDescriptor(opDesc, actDesc, CNNL_BIAS_ACTIVATION_GLU_ALGO_V1);

    cnnlBiasActivationGluForward_v2(handle, opDesc, inDesc, input,
                                    nullptr, nullptr, gateDesc, gate.data);

    cnrtFree(concatWorkspace);
    cnrtFree(input);
    cnnlDestroyActivationDescriptor(actDesc);
    cnnlDestroyBiasActivationGluDescriptor(opDesc);
    cnnlDestroyTensorDescriptor(gateDesc);
    cnnlDestroyTensorDescriptor(inDesc);
}
