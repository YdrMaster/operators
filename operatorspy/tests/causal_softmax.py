from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
import ctypes
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    CTensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
import torch


class CausalSoftmaxDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopCausalSoftmaxDescriptor_t = POINTER(CausalSoftmaxDescriptor)


def causal_softmax(x):
    type = x.dtype
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    y = x.clone()
    masked = torch.where(mask == 1, -torch.inf, y.to(torch.float32))
    return torch.nn.functional.softmax(masked, dim=-1).to(type)


def test(lib, handle, torch_device):
    x = torch.rand((32, 20, 512), dtype=torch.float16).to(torch_device)
    ans = causal_softmax(x)
    x_tensor = to_tensor(x, lib)
    descriptor = infiniopCausalSoftmaxDescriptor_t()
    check_error(
        lib.infiniopCreateCausalSoftmaxDescriptor(
            handle, ctypes.byref(descriptor), x_tensor.descriptor
        )
    )
    lib.infiniopCausalSoftmax(descriptor, None, 0, x_tensor.data, None)
    assert torch.allclose(x, ans, atol=0, rtol=1e-3)
    print("Test passed!")
    check_error(lib.infiniopDestroyCausalSoftmaxDescriptor(descriptor))


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    test(lib, handle, "cpu")
    destroy_handle(lib, handle)


def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    test(lib, handle, "cuda")
    destroy_handle(lib, handle)


def test_bang(lib):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    test(lib, handle, "mlu")
    destroy_handle(lib, handle)


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateCausalSoftmaxDescriptor.restype = c_int32
    lib.infiniopCreateCausalSoftmaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopCausalSoftmaxDescriptor_t),
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetCausalSoftmaxWorkspaceSize.restype = c_int32
    lib.infiniopGetCausalSoftmaxWorkspaceSize.argtypes = [
        infiniopCausalSoftmaxDescriptor_t,
        POINTER(c_uint64),
    ]
    lib.infiniopCausalSoftmax.restype = c_int32
    lib.infiniopCausalSoftmax.argtypes = [
        infiniopCausalSoftmaxDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyCausalSoftmaxDescriptor.restype = c_int32
    lib.infiniopDestroyCausalSoftmaxDescriptor.argtypes = [
        infiniopCausalSoftmaxDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib)
    if args.cuda:
        test_cuda(lib)
    if args.bang:
        test_bang(lib)
