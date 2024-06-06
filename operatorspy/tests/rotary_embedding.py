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

optype = OptypeEnum.OpRotaryEmbedding

Fn = ctypes.CFUNCTYPE(None, POINTER(Kernel), MutableTensor, ConstTensor, c_float)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def rotary_embedding(t, pos, theta, torch_device):
    dh = t.shape[2]
    freqs = (1.0 / (theta ** (torch.arange(0, dh, 2)[: (dh // 2)].float() / dh))).to(
        torch_device
    )
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    t_ = torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, t_)
    t_out = torch.view_as_real(t_ * freqs_cis).flatten(2).to(t.dtype)
    return t_out


def test(lib, device, config, rt_ctx_p, torch_device):
    op = lib.op_create(device, optype, config)
    kn = lib.kn_load(op, rt_ctx_p)
    fn = lib.fn_get(kn)
    fn = ctypes.cast(fn, Fn)
    t = torch.rand((4, 2, 2), dtype=torch.float16).to(torch_device)
    pos = torch.ones((4,), dtype=torch.int32).to(torch_device)
    theta = 1e4

    ans = rotary_embedding(t, pos, theta, torch_device)
    fn(kn, to_tensor(t), to_tensor(pos, False), theta)
    lib.kn_unload(kn)

    assert torch.allclose(t, ans, atol=1, rtol=1e-3)
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
    lib = open_lib("/home/duanchenjie/workspace/operators/build/linux/x86_64/release")
    test_cpu(lib)
    test_cuda(lib)
