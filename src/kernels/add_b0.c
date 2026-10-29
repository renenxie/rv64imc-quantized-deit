// add_b0.c — 殘差相加（B0），與 TFLite int8 ADD 逐位元相同
#include "ops.h"
#include "requant.h"

void qadd_b0(const qadd_params_t *p, const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int32_t x1 = ((int32_t)a[i] - p->zp1) * (1 << p->left_shift);
        int32_t x2 = ((int32_t)b[i] - p->zp2) * (1 << p->left_shift);
        int32_t v1 = multiply_by_quantized_multiplier(x1, p->m1, p->s1);
        int32_t v2 = multiply_by_quantized_multiplier(x2, p->m2, p->s2);
        out[i] = requantize(v1 + v2, p->m_out, p->s_out, p->out_zp, p->act_min, p->act_max);
    }
}
