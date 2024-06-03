import os
import ctypes
from ctypes import c_void_p, c_int, c_int64, c_uint64, Structure, POINTER

Device = c_int
Optype = c_int


class DataLayout(ctypes.Structure):
    _fields_ = [
        ("mantissa", ctypes.c_ushort),
        ("exponent", ctypes.c_ushort),
        ("sign", ctypes.c_ushort),
    ]

I8 = DataLayout(7, 0, 1)
I16 = DataLayout(15, 0, 1)
I32 = DataLayout(31, 0, 1)
I64 = DataLayout(63, 0, 1)
U8 = DataLayout(8, 0, 0)
U16 = DataLayout(16, 0, 0)
U32 = DataLayout(32, 0, 0)
U64 = DataLayout(64, 0, 0)
F16 = DataLayout(10, 5, 1)
BF16 = DataLayout(7, 8, 1)
F32 = DataLayout(23, 8, 1)
F64 = DataLayout(52, 11, 1)


class TensorLayout(Structure):
    _fields_ = [
        ("ndim", c_uint64),
        ("shape", POINTER(c_uint64)),
        ("pattern", POINTER(c_int64)),
    ]


class ConstTensor(Structure):
    _fields_ = [("layout", TensorLayout), ("data", c_void_p)]

class MutableTensor(Structure):
    _fields_ = [("layout", TensorLayout), ("data", c_void_p)]


class Kernel(Structure):
    pass

class Operator(Structure):
    pass

Op = POINTER(Operator)
Kn = POINTER(Kernel)

# Define Operator structure with function pointers
Operator._fields_ = [
    ("device", Device),
    ("optype", Optype),
    ("config", c_void_p),
    ("load", ctypes.CFUNCTYPE(Kn, POINTER(Operator), c_void_p)),
    ("drop", ctypes.CFUNCTYPE(None, POINTER(Operator))),
]

# Define Kernel structure with function pointers
Kernel._fields_ = [
    ("device", Device),
    ("optype", Optype),
    ("rt_ctx", c_void_p),
    ("fn", c_void_p),
    ("drop", ctypes.CFUNCTYPE(None, POINTER(Kernel))),
]

# Open operators library given directory path
def open_lib(library_path):
    # Load the library
    lib = ctypes.CDLL(os.path.join(library_path, "liboperators.so"))
    # Define functions
    lib.op_create.argtypes = [Device, Optype, c_void_p]
    lib.op_create.restype = Op

    lib.op_destroy.argtypes = [Op]
    lib.op_destroy.restype = None

    lib.kn_load.argtypes = [Op, c_void_p]
    lib.kn_load.restype = Kn

    lib.kn_unload.argtypes = [Kn]
    lib.kn_unload.restype = None

    lib.fn_get.argtypes = [Kn]
    lib.fn_get.restype = c_void_p
    return lib

# Convert PyTorch tensor to ConstTensor or MutableTensor
def to_tensor(tensor, mutable=True):
    # Get the number of dimensions
    ndim = tensor.ndimension()
    # Convert shape to ctypes array
    shape = (ctypes.c_uint64 * ndim)(*tensor.shape)
    # Create pattern (strides in bytes)

    pattern = (ctypes.c_int64 * ndim)(
        *(s * tensor.element_size() for s in tensor.stride())
    )
    # Get a pointer to the data
    data_ptr = tensor.data_ptr()
    # Create TensorLayout
    layout = TensorLayout(ndim, shape, pattern)
    # Create MutTensor
    if mutable:
        return MutableTensor(layout, ctypes.c_void_p(data_ptr))
    return ConstTensor(layout, ctypes.c_void_p(data_ptr))
