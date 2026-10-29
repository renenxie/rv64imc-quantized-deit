// matmul_b0.c — Attention 矩陣乘法（B0）：兩個輸入都有 zero point
#include "ops.h"
#include "requant.h"

void qmatmul_b0(const qmatmul_params_t *p, const int8_t *a, const int8_t *b, int8_t *out) {
    const int M = p->M, K = p->K, N = p->N;
    for (int m = 0; m < M; m++) {
        const int8_t *ar = a + m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *br = b + n * K;
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += ((int32_t)ar[k] - p->a_zp) * ((int32_t)br[k] - p->b_zp);
            out[m * N + n] = requantize(acc, p->quant_mult, p->shift,
                                        p->out_zp, p->act_min, p->act_max);
        }
    }
}
