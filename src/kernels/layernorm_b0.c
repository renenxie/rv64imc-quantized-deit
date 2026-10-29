// layernorm_b0.c — LayerNorm（B0）：整數平均、變異數、開根號
#include "ops.h"

// floor(sqrt(n))，逐位元法，只用加減與位移
static uint64_t isqrt64(uint64_t n) {
    uint64_t res = 0, bit = 1ULL << 62;
    while (bit > n) bit >>= 2;
    while (bit) {
        if (n >= res + bit) {
            n -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    return res;
}

// 四捨五入除法（b > 0）
static inline int64_t round_div(int64_t a, int64_t b) {
    return a >= 0 ? (a + b / 2) / b : -((-a + b / 2) / b);
}

void qlayernorm_b0(const qlayernorm_params_t *p, const int8_t *in, int8_t *out, int rows) {
    const int C = p->C;
    for (int r = 0; r < rows; r++) {
        const int8_t *x = in + r * C;
        int8_t *y = out + r * C;

        // ① Σ(x − zp)
        int32_t s = 0;
        for (int c = 0; c < C; c++) s += (int32_t)x[c] - p->in_zp;

        // ② V = Σ D²，D = C·(x − zp) − s = C·(x − mean)
        int64_t V = 0;
        for (int c = 0; c < C; c++) {
            int64_t d = (int64_t)C * ((int32_t)x[c] - p->in_zp) - s;
            V += d * d;
        }

        // ③ S ≈ C·std
        int64_t S = (int64_t)isqrt64((uint64_t)(V / C + p->eps));
        if (S == 0) S = 1;

        // ④ y = D·γ/std + β，再量化成 int8
        for (int c = 0; c < C; c++) {
            int64_t d = (int64_t)C * ((int32_t)x[c] - p->in_zp) - s;
            int64_t t = round_div(d * p->G[c], S) + p->B[c];
            int64_t q = ((t + (1LL << (QLN_F - 1))) >> QLN_F) + p->out_zp;
            y[c] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
        }
    }
}
