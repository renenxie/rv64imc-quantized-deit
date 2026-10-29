// ops.h — DeiT-Tiny 除了 FC 以外的運算（B0 baseline 介面）
//
// 全部只用整數運算；需要浮點的部分（查表內容、γ/β 的定點值、requant 參數）都在
// 離線（Python）先算好，推論時只用整數。
#ifndef OPS_H
#define OPS_H

#include <stdint.h>

// ---------------------------------------------------------------- GELU
// 輸入 int8 只有 256 種值 → 離線把 GELU 算好存成表，推論時查表。
//   out[i] = lut[in[i] + 128]
void qgelu_b0(const int8_t *in, int8_t *out, int n, const int8_t *lut);

// ---------------------------------------------------------------- 殘差相加
// 兩個 scale 不同的 int8 tensor 相加（TFLite ADD 語意）：
//   各自減 zero point、左移 left_shift 保留精度、各自 requant 到共同尺度、
//   相加後再 requant 到輸出尺度。
typedef struct {
    int32_t zp1, zp2, out_zp;
    int32_t left_shift;
    int32_t m1, m2, m_out;   // quant_mult
    int32_t s1, s2, s_out;   // shift
    int32_t act_min, act_max;
} qadd_params_t;

void qadd_b0(const qadd_params_t *p, const int8_t *a, const int8_t *b, int8_t *out, int n);

// ---------------------------------------------------------------- Softmax
// 每列：d = max − x（0~255）→ e = explut[d]（Q16 定點的 exp(−d·scale)）
//       → out = round(e·256 / Σe) − 128，輸出 scale = 1/256、zero point = −128
void qsoftmax_b0(const int8_t *in, int8_t *out, int rows, int L, const uint32_t *explut);

// ---------------------------------------------------------------- LayerNorm
// 每列（C 個 channel）：
//   D_c = C·(x_c − zp) − Σ(x − zp)          = C·(x − mean)，精確整數
//   S   = isqrt( Σ D² / C + eps )            ≈ C·std
//   y_c = round( D_c·G_c / S ) + B_c         （G、B 為 γ/s_out、β/s_out 的 Q16 定點值）
//   out = round(y_c / 2^16) + out_zp，飽和到 int8
typedef struct {
    int32_t C;
    int32_t in_zp, out_zp;
    int64_t eps;             // eps·C² / s_in²，離線算好
    const int32_t *G;        // [C]
    const int32_t *B;        // [C]
} qlayernorm_params_t;

#define QLN_F 16

void qlayernorm_b0(const qlayernorm_params_t *p, const int8_t *in, int8_t *out, int rows);

// ---------------------------------------------------------------- Attention 矩陣乘法
// out[m][n] = requant( Σ_k (a[m][k] − a_zp)·(b[n][k] − b_zp) )
// 兩個輸入都是 activation（都有 zero point），per-tensor requant、沒有 bias。
//   Q·Kᵀ：a = Q [tokens][64]，b = K [tokens][64]
//   P·V ：a = P [tokens][tokens]，b = Vᵀ [64][tokens]（V 需轉置存放，讓 K 維連續）
typedef struct {
    int M, K, N;
    int32_t a_zp, b_zp, out_zp;
    int32_t quant_mult, shift;
    int32_t act_min, act_max;
} qmatmul_params_t;

void qmatmul_b0(const qmatmul_params_t *p, const int8_t *a, const int8_t *b, int8_t *out);

#endif
