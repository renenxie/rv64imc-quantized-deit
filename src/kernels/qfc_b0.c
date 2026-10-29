// qfc_b0.c — B0 baseline：最直接的三層迴圈
// 刻意不做任何最佳化（不展開、不打包、zero point 在內迴圈減），
// 作為 B0 → B1 → X1 比較的起點。
#include "qfc.h"
#include "requant.h"

void qfc_b0(const qfc_params_t *p, const int8_t *in, const int8_t *w,
            const int32_t *bias, int8_t *out) {
    const int M = p->M, K = p->K, N = p->N;
    for (int m = 0; m < M; m++) {
        const int8_t *a = in + m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *wr = w + n * K;
            int32_t acc = bias[n];
            for (int k = 0; k < K; k++) {
                acc += ((int32_t)a[k] - p->in_zp) * (int32_t)wr[k];
            }
            out[m * N + n] = requantize(acc, p->quant_mult[n], p->shift[n],
                                        p->out_zp, p->act_min, p->act_max);
        }
    }
}
