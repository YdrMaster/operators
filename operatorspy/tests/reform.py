import ctypes
from ctypes import c_float, POINTER, c_void_p
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    MutableTensor,
    ConstTensor,
    DeviceEnum,
)

from operatorspy.tests.test_utils import get_args
import torch
import time


def test(lib, descriptor, torch_device):
    x = torch.rand((120, 10, 12, 32, 128), dtype=torch.float16).to(torch_device)
    y = torch.zeros_like(x)

    start = time.time()
    lib.reform(descriptor, to_tensor(y), to_tensor(x, False), None)
    end = time.time()
    print(f"Time elapsed: {(end - start) *1000} ms")

    assert torch.allclose(y, x, atol=1, rtol=1e-3)
    print("Test passed!")


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    config = None
    descriptor = lib.createReformDescriptor(device, config)
    test(lib, descriptor, "cpu")
    lib.destroyReformDescriptor(descriptor)


def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA
    config = None
    descriptor = lib.createReformDescriptor(device, config)
    test(lib, descriptor, "cuda")
    lib.destroyReformDescriptor(descriptor)


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.createReformDescriptor.restype = c_void_p
    lib.destroyReformDescriptor.argtypes = [c_void_p]
    lib.reform.argtypes = [
        c_void_p,
        MutableTensor,
        ConstTensor,
        c_void_p,
    ]
    if args.cpu:
        test_cpu(lib)
    if args.cuda:
        test_cuda(lib)
