import ctypes
from .data_layout import *
from .liboperators import infiniopTensorDescriptor_t, CTensor, infiniopHandle_t


def check_error(status):
    if status != 0:
        raise Exception("Error code " + str(status))


# Convert PyTorch tensor to library Tensor
def to_tensor(tensor, lib, shape=None, strides=None):
    import torch

    ndim = tensor.ndimension()
    if shape is None:
        shape = (ctypes.c_uint64 * ndim)(*tensor.shape)
    else:
        shape = (ctypes.c_uint64 * ndim)(*shape)
    # Get strides in bytes
    if strides is None:
        strides = (ctypes.c_int64 * ndim)(*(tensor.stride()))
    else:
        strides = (ctypes.c_int64 * ndim)(*strides)
    data_ptr = tensor.data_ptr()
    # fmt: off
    dt = (
        I8 if tensor.dtype == torch.int8 else
        I16 if tensor.dtype == torch.int16 else
        I32 if tensor.dtype == torch.int32 else
        I64 if tensor.dtype == torch.int64 else
        U8 if tensor.dtype == torch.uint8 else
        F16 if tensor.dtype == torch.float16 else
        BF16 if tensor.dtype == torch.bfloat16 else
        F32 if tensor.dtype == torch.float32 else
        F64 if tensor.dtype == torch.float64 else
        # TODO: These following types may not be supported by older 
        # versions of PyTorch.
        U16 if tensor.dtype == torch.uint16 else
        U32 if tensor.dtype == torch.uint32 else
        U64 if tensor.dtype == torch.uint64 else
        None
    )
    # fmt: on
    assert dt is not None
    # Create TensorDecriptor
    tensor_desc = infiniopTensorDescriptor_t()
    lib.infiniopCreateTensorDescriptor(
        ctypes.byref(tensor_desc), ndim, shape, strides, dt
    )
    # Create Tensor
    return CTensor(tensor_desc, data_ptr)


def create_handle(lib, device, id=0):
    handle = infiniopHandle_t()
    check_error(lib.infiniopCreateHandle(ctypes.byref(handle), device, id))
    return handle


def destroy_handle(lib, handle):
    check_error(lib.infiniopDestroyHandle(handle))
