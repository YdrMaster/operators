import ctypes

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
