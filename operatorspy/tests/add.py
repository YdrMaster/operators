from ctypes import POINTER, Structure, c_int32, c_uint16, c_uint64, c_void_p
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
)

from operatorspy.tests.test_utils import get_args, Inplace
import torch


class AddDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopAddDescriptor_t = POINTER(AddDescriptor)


def add(x, y):
    return x + y


def test(
    lib,
    handle,
    torch_device,
    tensor_shape,
    tensor_stride=None,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Add on {torch_device} with tensor_shape:{tensor_shape} tensor_stride:{tensor_stride} dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    if torch_device == "cuda" and inplace == Inplace.INPLACE_B:
        print("Unsupported test: CUDA does not support inplace b")
        return

    a = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device)
    b = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device)
    c = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else (a if inplace == Inplace.INPLACE_A else b)

    ans = add(a, b)

    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    c_tensor = to_tensor(c, lib) if inplace == Inplace.OUT_OF_PLACE else (a_tensor if inplace == Inplace.INPLACE_A else b_tensor)
    descriptor = infiniopAddDescriptor_t()

    check_error(
        lib.infiniopCreateAddDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )
    lib.infiniopAdd(
        descriptor, None, 0, c_tensor.data, a_tensor.data, b_tensor.data, None
    )
    assert torch.allclose(c, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyAddDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, x_stride, inplace in test_cases:
        test(lib, handle, "cpu", x_shape, x_stride, inplace=inplace)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, x_stride, inplace in test_cases:
        test(lib, handle, "cuda", x_shape, x_stride, inplace=inplace)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for x_shape, x_stride in test_cases:
        test(lib, handle, "mlu", x_shape, x_stride)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, x_stride, inplace
        ((32, 20, 512), None, Inplace.OUT_OF_PLACE),
        ((32, 20, 512), None, Inplace.INPLACE_A),
        ((32, 20, 512), None, Inplace.INPLACE_B),
        ((32), None, Inplace.OUT_OF_PLACE),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateAddDescriptor.restype = c_int32
    lib.infiniopCreateAddDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopAddDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopAdd.restype = c_int32
    lib.infiniopAdd.argtypes = [
        infiniopAddDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyAddDescriptor.restype = c_int32
    lib.infiniopDestroyAddDescriptor.argtypes = [
        infiniopAddDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    print("Test passed!")
