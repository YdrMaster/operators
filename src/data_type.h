#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include <stdint.h>

struct DataLayout {
    int mantissa : 16,
        exponent : 15,
        sign : 1;
};


// clang-format off
const static struct DataLayout
    I8   = { 7,  0, 1},
    I16  = {15,  0, 1},
    I32  = {31,  0, 1},
    I64  = {63,  0, 1},
    U8   = { 8,  0, 0},
    U16  = {16,  0, 0},
    U32  = {32,  0, 0},
    U64  = {64,  0, 0},
    F16  = {10,  5, 1},
    BF16 = { 7,  8, 1},
    F32  = {23,  8, 1},
    F64  = {52, 11, 1};
// clang-format on

#endif// __DATA_TYPE_H__
