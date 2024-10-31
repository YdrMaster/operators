#ifndef __COMMON_CPU_H__
#define __COMMON_CPU_H__

#include <cmath>
#include <cstdint>

// return a mask with the specified number of low bits set to 1
constexpr static uint16_t mask_low(int bits) noexcept {
    return (1 << bits) - 1;
}

// convert half-precision float to single-precision float
float f16_to_f32(uint16_t code);

// convert single-precision float to half-precision float
uint16_t f32_to_f16(float val);

// get the corresponding index in the destination given the flat index of the source
uint64_t getDstIndex(uint64_t flat_index, uint64_t ndim, int64_t const *src_strides, int64_t const *dst_strides);

// get the offset of the next element in a tensor given its flat index
uint64_t getNextIndex(uint64_t flat_index, uint64_t ndim, uint64_t const *shape, int64_t const *strides);

#endif// __COMMON_CPU_H__
