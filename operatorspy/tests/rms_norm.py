import ctypes
from .. import (
    open_lib,
    DeviceEnum,
    OptypeEnum,
    to_tensor,
    MutableTensor,
    ConstTensor,
    Kn,
)
import torch

optype = OptypeEnum.OpRmsNorm
Fn = ctypes.CFUNCTYPE(None, Kn, MutableTensor, ConstTensor, ConstTensor, ctypes.c_float)


def rms_norm(x, w, eps):
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return w * hidden_states.to(input_dtype)


def test_cpu(lib):
    device = DeviceEnum.DEVICE_CPU
    config = None
    op = lib.op_create(device, optype, config)
    rt_ctx = None
    kn = lib.kn_load(op, rt_ctx)
    fn = lib.fn_get(kn)
    fn = ctypes.cast(fn, Fn)

    import torch

    y = torch.zeros((5, 16), dtype=torch.float16)
    x = torch.rand((5, 16), dtype=torch.float16)
    w = torch.ones((16,), dtype=torch.float16)
    eps = 1e-5
    fn(kn, to_tensor(y), to_tensor(x, False), to_tensor(w, False), eps)
    lib.kn_unload(kn)

    ans = rms_norm(x, w, eps)

    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    print("Test passed!")


if __name__ == "__main__":
    lib = open_lib("/data1/shared/panzezhong/operators/build/linux/x86_64/release")
    test_cpu(lib)
