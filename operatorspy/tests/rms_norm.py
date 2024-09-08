from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float
import ctypes
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
    rearrange_tensor,
    create_workspace,
)

from operatorspy.tests.test_utils import get_args
import torch


class RMSNormDescriptor(Structure):
    _fields_ = [("device", c_int32)]
    
infiniopRMSNormDescriptor_t = POINTER(RMSNormDescriptor)


def rms_norm(x, w, eps):
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return w * hidden_states.to(input_dtype)


def test(lib, handle, torch_device, yshape, xshape, wshape, _dtype):
    print(
        f"Testing RMSNorm on {torch_device} with x_shape:{xshape}, y_shape:{yshape}, w_shape:{wshape}, dtype:{_dtype}."
    )
    # import pdb; pdb.set_trace();
    y = torch.zeros(yshape, dtype=_dtype).to(torch_device)
    x = torch.rand(xshape, dtype=_dtype).to(torch_device)
    w = torch.ones(wshape, dtype=_dtype).to(torch_device)
    eps = 1e-5
    ans = rms_norm(x, w, eps)
    
    y_tensor = to_tensor(y, lib)
    x_tensor = to_tensor(x, lib)
    w_tensor = to_tensor(w, lib)
    descriptor = infiniopRMSNormDescriptor_t()
    check_error(
        lib.infiniopCreateRMSNormDescriptor(
            handle, 
            ctypes.byref(descriptor), 
            y_tensor.descriptor, 
            x_tensor.descriptor, 
            w_tensor.descriptor, 
            c_float(eps)
        )
    )
    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetRMSNormWorkspaceSize(
            descriptor,
            ctypes.byref(workspace_size)
        )
    )
    workspace = to_tensor(create_workspace(workspace_size.value, y.device), lib)
    check_error(
        lib.infiniopRMSNorm(
            descriptor,
            workspace.data if workspace is not None else None,
            workspace_size.value,
            y_tensor.data,
            x_tensor.data,
            w_tensor.data,
            None
        )
    )

    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    print("Test passed!")


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    descriptor = lib.createRMSNormDescriptor(device, None)
    test(lib, descriptor, "cpu")
    lib.destroyRMSNormDescriptor(descriptor)


def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA
    descriptor = lib.createRMSNormDescriptor(device, None)
    test(lib, descriptor, "cuda")
    lib.destroyRMSNormDescriptor(descriptor)

def test_bang(lib):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    descriptor = lib.createRMSNormDescriptor(device, None)
    test(lib, descriptor, "mlu")
    lib.destroyRMSNormDescriptor(descriptor)


def test_ascend(lib, test_cases):
    import torch_npu
    device = DeviceEnum.DEVICE_NPU
    handle = create_handle(lib, device)
    for yshape, xshape, wshape, dtype in test_cases:
        test(lib, handle, "npu", yshape, xshape, wshape, dtype)
    
    destroy_handle(lib, handle)
    


if __name__ == "__main__":
    test_cases = [
        ((16, 2048), (16, 2048), (2048,), torch.float16),
        ((16, 2048), (16, 2048), (2048,), torch.float32),
    ]
    
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateRMSNormDescriptor.restype = c_int32
    lib.infiniopCreateRMSNormDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopRMSNormDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float
    ]
    
    lib.infiniopGetRMSNormWorkspaceSize.restype = c_int32
    lib.infiniopGetRMSNormWorkspaceSize.argtypes = [
        infiniopRMSNormDescriptor_t,
        POINTER(c_uint64)
    ]
    
    lib.infiniopRMSNorm.restype = c_int32
    lib.infiniopRMSNorm.argtypes = [
        infiniopRMSNormDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p
    ]
    
    lib.infiniopDestroyRMSNormDescriptor.restype = c_int32
    lib.infiniopDestroyRMSNormDescriptor.argtypes = [
        infiniopRMSNormDescriptor_t,
    ]
    
    if args.cpu:
        test_cpu(lib)
    if args.cuda:
        test_cuda(lib)
    if args.bang:
        test_bang(lib)
    if args.ascend:
        test_ascend(lib, test_cases)
