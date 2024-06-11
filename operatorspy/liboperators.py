import os
import ctypes
from ctypes import c_void_p, c_int, c_int64, c_uint64, Structure, POINTER
from .data_layout import *

Device = c_int
Optype = c_int


class TensorLayout(Structure):
    _fields_ = [
        ("dt", DataLayout),
        ("ndim", c_uint64),
        ("offset", c_uint64),
        ("shape", POINTER(c_uint64)),
        ("pattern", POINTER(c_int64)),
    ]


class ConstTensor(Structure):
    _fields_ = [("layout", TensorLayout), ("data", c_void_p)]


class MutableTensor(Structure):
    _fields_ = [("layout", TensorLayout), ("data", c_void_p)]


# Open operators library
def open_lib():
    def find_library_in_ld_path(library_name):
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        paths = ld_library_path.split(os.pathsep)
        for path in paths:
            full_path = os.path.join(path, library_name)
            if os.path.isfile(full_path):
                return full_path
        return None

    # Load the library
    library_path = find_library_in_ld_path("liboperators.so")
    assert library_path is not None, "Cannot find liboperators.so in LD_LIBRARY_PATH"
    lib = ctypes.CDLL(library_path)
    return lib


# Convert PyTorch tensor to ConstTensor or MutableTensor
def to_tensor(tensor, mutable=True):
    import torch

    ndim = tensor.ndimension()
    shape = (ctypes.c_uint64 * ndim)(*tensor.shape)
    # Get strides in bytes
    strides = (ctypes.c_int64 * ndim)(
        *(s * tensor.element_size() for s in tensor.stride())
    )
    data_ptr = tensor.data_ptr()
    dt = (
        I8
        if tensor.dtype == torch.int8
        else (
            I16
            if tensor.dtype == torch.int16
            else (
                I32
                if tensor.dtype == torch.int32
                else (
                    I64
                    if tensor.dtype == torch.int64
                    else (
                        U8
                        if tensor.dtype == torch.uint8
                        # TODO: Some PyTorch dtypes are not supported yet
                        # else U16 if tensor.dtype == torch.uint16
                        # else U32 if tensor.dtype == torch.uint32
                        # else U64 if tensor.dtype == torch.uint64
                        else (
                            F16
                            if tensor.dtype == torch.float16
                            else (
                                BF16
                                if tensor.dtype == torch.bfloat16
                                else (
                                    F32
                                    if tensor.dtype == torch.float32
                                    else F64 if tensor.dtype == torch.float64 else None
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    assert dt is not None
    # Create TensorLayout
    layout = TensorLayout(dt, ndim, 0, shape, strides)
    # Create MutTensor
    if mutable:
        return MutableTensor(layout, ctypes.c_void_p(data_ptr))
    return ConstTensor(layout, ctypes.c_void_p(data_ptr))
