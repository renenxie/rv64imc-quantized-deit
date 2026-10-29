// softmax_b0.c — Softmax（B0）：exp 查表 + 每個元素一次整數除法
#include "ops.h"

void qsoftmax_b0(const int8_t *in, int8_t *out, int rows, int L, const uint32_t *explut) {
    for (int r = 0; r < rows; r++) {
        const int8_t *x = in + r * L;
        int8_t *y = out + r * L;

        // ① 找最大值（減掉最大值後，exp 的輸入固定在 0~255，才能查表）
        int32_t mx = x[0];
        for (int i = 1; i < L; i++)
            if (x[i] > mx) mx = x[i];

        // ② Σ exp
        uint32_t sum = 0;
        for (int i = 0; i < L; i++)
            sum += explut[mx - x[i]];

        // ③ 正規化：round(e·256 / sum) − 128
        for (int i = 0; i < L; i++) {
            uint32_t e = explut[mx - x[i]];
            int32_t q = (int32_t)((((uint64_t)e << 8) + sum / 2) / sum) - 128;
            y[i] = (int8_t)(q > 127 ? 127 : q);
        }
    }
}
