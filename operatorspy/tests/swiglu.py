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
    Kernel,
    DeviceEnum,
    OptypeEnum,
)

from operatorspy.tests.test_utils import get_args
import torch

optype = OptypeEnum.OpSwiglu

Fn = ctypes.CFUNCTYPE(None, POINTER(Kernel), MutableTensor, ConstTensor)


def swiglu(gate, up):
    return up * torch.nn.functional.silu(gate).to(gate.dtype)


def test(lib, device, config, rt_ctx_p, torch_device):
    op = lib.op_create(device, optype, config)
    kn = lib.kn_load(op, rt_ctx_p)
    fn = lib.fn_get(kn)
    fn = ctypes.cast(fn, Fn)
    gate = torch.rand((1, 64), dtype=torch.float16).to(torch_device)
    up = torch.rand((1, 64), dtype=torch.float16).to(torch_device)

    ans = swiglu(gate, up)
    fn(kn, to_tensor(gate), to_tensor(up, False))
    lib.kn_unload(kn)

    assert torch.allclose(gate, ans, atol=2, rtol=2)
    print("Test passed!")


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    config = None
    rt_ctx_p = None
    test(lib, device, config, rt_ctx_p, "cpu")


def test_cuda(lib):
    device = DeviceEnum.DEVICE_CUDA

    class CudaContext(ctypes.Structure):
        _fields_ = [("stream", c_void_p)]

    config = None
    rt_ctx_p = ctypes.byref(CudaContext(None))

    test(lib, device, config, rt_ctx_p, "cuda")


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    if args.cpu:
        test_cpu(lib)
    if args.cuda:
        test_cuda(lib)
