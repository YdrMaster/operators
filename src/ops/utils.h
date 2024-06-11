#ifndef __UTILS_H__
#define __UTILS_H__

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdio>  
#include <cstdlib>

/* This file contains some useful macros and helper functions */

// check if an expression is true, and if not, print an error message and abort the program
#define MAX_BACKTRACE_DEPTH 10

inline void assert_true(int expr, const char *msg, const char *file, int line) {
    if (!expr) {
        fprintf(stderr, "\033[31mAssertion failed:\033[0m %s at file %s, line %d\n", msg, file, line);

        void *array[MAX_BACKTRACE_DEPTH];
        size_t size = backtrace(array, MAX_BACKTRACE_DEPTH);
        fprintf(stderr, "    Stack trace (most recent call first):\n");
        backtrace_symbols_fd(array, size, STDERR_FILENO);

        exit(EXIT_FAILURE);
    }
}


#define ASSERT(expr) assert_true(expr, #expr " is false", __FILE__, __LINE__)
#define ASSERT_EQ(a, b) assert_true((a) == (b), #a " != "#b, __FILE__, __LINE__)
#define ASSERT_VALID_PTR(a) assert_true((a)!= nullptr, #a " is nullptr",__FILE__, __LINE__)

#define PANIC(EXPR)                                             \
    printf("Error at %s:%d - %s\n", __FILE__, __LINE__, #EXPR); \
    exit(1)

#endif// __UTILS_H__
