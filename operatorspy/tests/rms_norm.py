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
import torch

optype = OptypeEnum.OpRmsNorm

Fn = ctypes.CFUNCTYPE(
    None, POINTER(Kernel), MutableTensor, ConstTensor, ConstTensor, c_float
)


def rms_norm(x, w, eps):
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return w * hidden_states.to(input_dtype)


def test(lib, device, config, rt_ctx_p, torch_device):
    op = lib.op_create(device, optype, config)
    kn = lib.kn_load(op, rt_ctx_p)
    fn = lib.fn_get(kn)
    fn = ctypes.cast(fn, Fn)
    y = torch.zeros((5, 16), dtype=torch.float16).to(torch_device)
    x = torch.rand((5, 16), dtype=torch.float16).to(torch_device)
    w = torch.ones((16,), dtype=torch.float16).to(torch_device)

    eps = 1e-5
    ans = rms_norm(x, w, eps)
    fn(kn, to_tensor(y), to_tensor(x, False), to_tensor(w, False), eps)
    lib.kn_unload(kn)

    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
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
    lib = open_lib("/data1/shared/panzezhong/operators/build/linux/x86_64/release")
    test_cuda(lib)
