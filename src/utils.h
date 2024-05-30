#ifndef __UTILS_H__
#define __UTILS_H__

/* This file contains some useful macros and helper functions */


#ifdef __cplusplus
extern "C" {
#endif

// check if an expression is true, and if not, print an error message and abort the program
void assert_true(int expr, const char * msg, const char *file, int line);

#ifdef __cplusplus
}
#endif

#define ASSERT(expr) assert_true(expr, #expr " is false", __FILE__, __LINE__)
#define ASSERT_EQ(a, b) assert_true((a) == (b), #a " != "#b, __FILE__, __LINE__)
#define ASSERT_VALID_PTR(a) assert_true((a)!= nullptr, #a " is nullptr",__FILE__, __LINE__)

#endif// __UTILS_H__
