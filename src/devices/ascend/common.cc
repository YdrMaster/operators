#include "common.h"
#include <cassert>
#include <vector>

int64_t *castToInt64_t(uint64_t *in, uint64_t cnt) {
    assert(in);

    std::vector<int64_t> out(cnt);
    for (size_t i = 0; i < cnt; ++i) {
        out[i] = static_cast<int64_t>(in[i]);
    }
    return out.data();
}

int64_t shapeProd(int64_t *in, uint64_t cnt) {
    int64_t prod = 1;
    for (size_t i = 0; i < cnt; i++) {
        prod *= in[i];
    }
    return prod;
}
