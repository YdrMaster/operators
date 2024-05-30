#include "utils.h"
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define MAX_BACKTRACE_DEPTH 10

void assert_true(int expr, const char *msg, const char *file, int line) {
    if (!expr) {
        fprintf(stderr, "\033[31mAssertion failed:\033[0m %s at file %s, line %d\n", msg, file, line);

        void *array[MAX_BACKTRACE_DEPTH];
        size_t size = backtrace(array, MAX_BACKTRACE_DEPTH);
        fprintf(stderr, "    Stack trace (most recent call first):\n");
        backtrace_symbols_fd(array, size, STDERR_FILENO);

        exit(EXIT_FAILURE);
    }
}
