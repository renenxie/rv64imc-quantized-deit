// requant.h — TFLite / LiteRT 相容的整數 requantization
//
// real_multiplier = in_scale * w_scale / out_scale  (0 < real < 1 通常成立)
// 以 (quant_mult, shift) 表示：real ≈ quant_mult * 2^(shift - 31)
//   quant_mult ∈ [2^30, 2^31)，shift ≤ 0 表示右移
//
// 流程：int32 累加值 → 定點乘法 → 捨入 → 位移 → + zero point → 飽和
// 這正是規格書 3 節要用 requant 指令加速的部分。
#ifndef REQUANT_H
#define REQUANT_H

#include <stdint.h>

// (a * b * 2) 的高 32 位元，含 round-to-nearest；對應 gemmlowp SaturatingRoundingDoublingHighMul
static inline int32_t sat_round_doubling_high_mul(int32_t a, int32_t b) {
    if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
    int64_t ab = (int64_t)a * (int64_t)b;
    int64_t nudge = ab >= 0 ? (1LL << 30) : (1 - (1LL << 30));
    // C 的整數除法向 0 截斷，與 gemmlowp 參考實作一致
    return (int32_t)((ab + nudge) / (1LL << 31));
}

// 除以 2^exponent，round-half-away-from-zero；對應 gemmlowp RoundingDivideByPOT
static inline int32_t rounding_divide_by_pot(int32_t x, int exponent) {
    int32_t mask = (int32_t)((1LL << exponent) - 1);
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

static inline int32_t multiply_by_quantized_multiplier(int32_t x, int32_t quant_mult, int shift) {
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    return rounding_divide_by_pot(
        sat_round_doubling_high_mul((int32_t)((uint32_t)x << left_shift), quant_mult),
        right_shift);
}

// 完整 requant：acc → int8
static inline int8_t requantize(int32_t acc, int32_t quant_mult, int shift,
                                int32_t out_zp, int32_t act_min, int32_t act_max) {
    int32_t v = multiply_by_quantized_multiplier(acc, quant_mult, shift) + out_zp;
    if (v < act_min) v = act_min;
    if (v > act_max) v = act_max;
    return (int8_t)v;
}

#endif
