import ctypes
from ctypes import c_ushort


class DataLayout(ctypes.LittleEndianStructure):
    _fields_ = [
        ("packed", c_ushort, 8),
        ("sign", c_ushort, 1),
        ("size", c_ushort, 7),
        ("mantissa", c_ushort, 8),
        ("exponent", c_ushort, 8),
    ]


I8 = DataLayout(1, 1, 1, 7, 0)
I16 = DataLayout(1, 1, 2, 15, 0)
I32 = DataLayout(1, 1, 4, 31, 0)
I64 = DataLayout(1, 1, 8, 63, 0)
U8 = DataLayout(1, 0, 1, 8, 0)
U16 = DataLayout(1, 0, 2, 16, 0)
U32 = DataLayout(1, 0, 4, 32, 0)
U64 = DataLayout(1, 0, 8, 64, 0)
F16 = DataLayout(1, 1, 2, 10, 5)
BF16 = DataLayout(1, 1, 2, 7, 8)
F32 = DataLayout(1, 1, 4, 23, 8)
F64 = DataLayout(1, 1, 8, 52, 11)
