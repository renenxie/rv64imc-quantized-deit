// qfc.h — 量化 Fully-Connected（= GEMM）介面
//
//   out[m][n] = requant( bias[n] + Σ_k (in[m][k] - in_zp) * w[n][k] )
//
// 版面配置（與 TFLite FC 相同）：
//   in   : [M][K] int8，逐列連續
//   w    : [N][K] int8，每個輸出通道的權重連續（對 Xqmac8 一次讀 8 個 int8 最有利）
//   bias : [N]    int32
//   out  : [M][N] int8
// 權重為對稱量化（zero point = 0），支援 per-channel requant。
#ifndef QFC_H
#define QFC_H

#include <stdint.h>

typedef struct {
    int M, K, N;
    int32_t in_zp;            // 輸入 zero point
    int32_t out_zp;           // 輸出 zero point
    int32_t act_min, act_max; // 輸出飽和範圍（含 ReLU 時 act_min = out_zp）
    const int32_t *quant_mult; // [N]
    const int8_t  *shift;      // [N]
} qfc_params_t;

// B0：直接、未最佳化的參考實作（baseline）
void qfc_b0(const qfc_params_t *p, const int8_t *in, const int8_t *w,
            const int32_t *bias, int8_t *out);

#endif
