#include "common_cpu.h"

float f16_to_f32(uint16_t code) {
    union {
        uint32_t u32;
        float f32;
    } ans{0};
    ans.u32 = ((code & 0x8000) << 16) |
              ((code & 0x7C00) == 0 ? 0 : (((code & 0x7C00) >> 10) + 112) << 23) |
              ((code & 0x03FF) << 13);
    return ans.f32;
}

uint16_t f32_to_f16(float val) {
    union {
        float f32;
        uint32_t u32;
    } x{val};
    return (static_cast<uint16_t>(x.u32 >> 16) & (1 << 15)) |
           (((x.u32 >> 23) & mask_low(8)) >= 112
                ? static_cast<uint16_t>(
                      std::min((x.u32 >> 23 & mask_low(8)) - 127 + 15,
                               static_cast<uint32_t>(31)))
                : 0)
               << 10 |
           static_cast<uint16_t>(x.u32 >> 13) & mask_low(10);
}