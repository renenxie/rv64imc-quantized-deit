// gelu_b0.c — GELU：查表（B0）
#include "ops.h"

void qgelu_b0(const int8_t *in, int8_t *out, int n, const int8_t *lut) {
    for (int i = 0; i < n; i++) {
        out[i] = lut[(int32_t)in[i] + 128];
    }
}
