#ifndef __UTILS_H__
#define __UTILS_H__

#include <cmath>
#include <cstdint>

/* This file contains some useful macros and helper functions */

// check if an expression is true, and if not, print an error message and abort the program
void assert_true(int expr, const char * msg, const char *file, int line);

// return a mask with the specified number of low bits set to 1
constexpr static uint16_t mask_low(int bits) noexcept {
    return (1 << bits) - 1;
}

// convert half-precision float to single-precision float
float f16_to_f32(uint16_t code);

// convert single-precision float to half-precision float
uint16_t f32_to_f16(float val);

#define ASSERT(expr) assert_true(expr, #expr " is false", __FILE__, __LINE__)
#define ASSERT_EQ(a, b) assert_true((a) == (b), #a " != "#b, __FILE__, __LINE__)
#define ASSERT_VALID_PTR(a) assert_true((a)!= nullptr, #a " is nullptr",__FILE__, __LINE__)

#endif// __UTILS_H__
