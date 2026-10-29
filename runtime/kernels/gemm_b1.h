// gemm_b1.h — B1 共用的 INT8 內積核心（只用 RV64IMC 原有指令）
//
//   C[i][j] = Σ_k A[i][k] · B[j][k]      A：M×K，B：N×K（兩者都是一列一列連續存放）
//
// FULLY_CONNECTED（A = 輸入 token、B = 權重）與 BATCH_MATMUL（A = Q 或 P、B = K 或 Vᵀ）共用。
// 算完每個 C[i][j] 後呼叫 finalize(i, j, acc)，由各 kernel 自己做 zero-point 修正與 requant。
//
// 優化：
//   ・暫存器分塊：一次 2 列 A × 4 列 B = 8 個累加器 → 每個 MAC 只需 0.75 次 load
//   ・指標 + 固定位移（lb rd, o(ptr)），內層展開 4 次 → 每 4 個 k 才遞增一次指標
// int32 累加不會溢位（|a|,|b| ≤ 128、K ≤ 768），加法順序不同不影響結果。
//
// 使用這個標頭檔的 .cc 要在 include 前加 #pragma GCC optimize("no-schedule-insns")：
// 否則編譯器的「暫存器配置前排程」會把 load 提前太多，暫存器不足而溢出到 stack。
#ifndef RUNTIME_KERNELS_GEMM_B1_H_
#define RUNTIME_KERNELS_GEMM_B1_H_

#include <cstdint>

namespace b1 {

template <typename Finalize>
inline void GemmInt8(const int8_t* __restrict A, const int8_t* __restrict B, int M, int N, int K,
                     Finalize finalize) {
  const int k4 = K & ~3;
  int i = 0;
  for (; i + 1 < M; i += 2) {  // 2 列 A
    const int8_t* a0 = A + i * K;
    const int8_t* a1 = a0 + K;
    int j = 0;
    for (; j + 3 < N; j += 4) {  // 4 列 B
      const int8_t* b0 = B + j * K;
      const int8_t* b1 = b0 + K;
      const int8_t* b2 = b1 + K;
      const int8_t* b3 = b2 + K;
      int32_t c00 = 0, c01 = 0, c02 = 0, c03 = 0;
      int32_t c10 = 0, c11 = 0, c12 = 0, c13 = 0;
      const int8_t* p0 = a0;
      const int8_t* p1 = a1;
      const int8_t* q0 = b0;
      const int8_t* q1 = b1;
      const int8_t* q2 = b2;
      const int8_t* q3 = b3;
      const int8_t* const p_end4 = a0 + k4;
#define B1_STEP_2x4(o)                                                \
  {                                                                   \
    const int32_t x0 = p0[o], x1 = p1[o];                             \
    const int32_t y0 = q0[o], y1 = q1[o], y2 = q2[o], y3 = q3[o];     \
    c00 += x0 * y0; c01 += x0 * y1; c02 += x0 * y2; c03 += x0 * y3;   \
    c10 += x1 * y0; c11 += x1 * y1; c12 += x1 * y2; c13 += x1 * y3;   \
  }
      while (p0 != p_end4) {
        B1_STEP_2x4(0) B1_STEP_2x4(1) B1_STEP_2x4(2) B1_STEP_2x4(3)
        p0 += 4; p1 += 4; q0 += 4; q1 += 4; q2 += 4; q3 += 4;
      }
      while (p0 != a0 + K) {  // K 不是 4 的倍數時的尾端
        B1_STEP_2x4(0)
        ++p0; ++p1; ++q0; ++q1; ++q2; ++q3;
      }
#undef B1_STEP_2x4
      finalize(i, j, c00); finalize(i, j + 1, c01); finalize(i, j + 2, c02); finalize(i, j + 3, c03);
      finalize(i + 1, j, c10); finalize(i + 1, j + 1, c11);
      finalize(i + 1, j + 2, c12); finalize(i + 1, j + 3, c13);
    }
    for (; j < N; ++j) {  // 剩下不足 4 列的 B
      const int8_t* bj = B + j * K;
      int32_t c0 = 0, c1 = 0;
      for (int k = 0; k < K; ++k) {
        c0 += a0[k] * bj[k];
        c1 += a1[k] * bj[k];
      }
      finalize(i, j, c0);
      finalize(i + 1, j, c1);
    }
  }
  for (; i < M; ++i) {  // 剩下的單一列 A：1 × 4
    const int8_t* a0 = A + i * K;
    int j = 0;
    for (; j + 3 < N; j += 4) {
      const int8_t* b0 = B + j * K;
      const int8_t* b1 = b0 + K;
      const int8_t* b2 = b1 + K;
      const int8_t* b3 = b2 + K;
      int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
      for (int k = 0; k < K; ++k) {
        const int32_t x0 = a0[k];
        c0 += x0 * b0[k]; c1 += x0 * b1[k]; c2 += x0 * b2[k]; c3 += x0 * b3[k];
      }
      finalize(i, j, c0); finalize(i, j + 1, c1); finalize(i, j + 2, c2); finalize(i, j + 3, c3);
    }
    for (; j < N; ++j) {
      const int8_t* bj = B + j * K;
      int32_t c0 = 0;
      for (int k = 0; k < K; ++k) c0 += a0[k] * bj[k];
      finalize(i, j, c0);
    }
  }
}

}  // namespace b1

#endif  // RUNTIME_KERNELS_GEMM_B1_H_
