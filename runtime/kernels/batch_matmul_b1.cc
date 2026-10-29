// batch_matmul_b1.cc — B1：軟體優化的 INT8 BATCH_MATMUL（只用 RV64IMC 原有指令）
//
// DeiT 的 attention 用到兩種：
//   Q·Kᵀ：lhs = Q [3, 197, 64]，rhs = K [3, 197, 64]，adj_y = true  → 197×197，K = 64
//   P·V ：lhs = P [3, 197, 197]，rhs = V [3, 197, 64]，adj_y = false → 197×64，K = 197
//
// 與 LiteRT Micro 的 reference kernel（B0）算出「逐位元相同」的結果，只改寫法：
//
//  1. 兩邊的 zero-point 都移出內層迴圈
//       B0：acc = Σ_k (a[i][k] − za)·(b[j][k] − zb)          每個 MAC 多兩個減法
//       B1：acc = Σ_k a·b  − zb·Σ_k a[i][k] − za·Σ_k b[j][k] + K·za·zb
//                          └─ 每列一次 ─┘   └─ 每行一次 ─┘    （每個 head 算一次，成本很小）
//  2. 內積核心與 FULLY_CONNECTED 共用（gemm_b1.h）：2 × 4 暫存器分塊、展開 4 次
//     → 順便消除 B0 內層迴圈每圈從 stack 重讀 2 次 zero-point 的問題
//  3. adj_y = false（P·V）時，V 先轉置成 [64][197]，讓內積兩邊都是連續記憶體（B0 也做同樣的轉置）
//
// requant 與 B0 相同：tflite::micro::MultiplyByQuantizedMultiplier（兩次四捨五入的版本）。
// 只處理 DeiT 用到的情況（INT8、adj_x = false、兩邊 batch 維度相同）；其他情況交回原本的 kernel。
#pragma GCC optimize("no-schedule-insns")  // 理由見 gemm_b1.h

#include <cstdint>

#include "kernels/requant_b1.h"

#include "kernels/gemm_b1.h"
#include "tensorflow/lite/micro/kernels/batch_matmul.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace {

using tflite::micro::OpDataBatchMatmul;

constexpr int kMaxDim = 256;  // 列和、行和的暫存陣列大小（DeiT 最大 197）

const TFLMRegistration& BaseRegistration() {
  static const TFLMRegistration reg = tflite::micro::Register_BATCH_MATMUL();
  return reg;
}

void* BmmB1Init(TfLiteContext* context, const char* buffer, size_t length) {
  return BaseRegistration().init(context, buffer, length);
}

TfLiteStatus BmmB1Prepare(TfLiteContext* context, TfLiteNode* node) {
  // LiteRT Micro 原本的 Prepare：檢查形狀、配置轉置用的暫存 tensor、requant 參數（離線參數表）
  return BaseRegistration().prepare(context, node);
}

TfLiteStatus BmmB1Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = static_cast<const TfLiteBatchMatMulParams*>(node->builtin_data);
  auto* data = static_cast<OpDataBatchMatmul*>(node->user_data);
  const TfLiteEvalTensor* lhs = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* rhs = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  const tflite::RuntimeShape ls = tflite::micro::GetTensorShape(lhs);
  const tflite::RuntimeShape rs = tflite::micro::GetTensorShape(rhs);
  const int rank = ls.DimensionsCount();
  bool fast = lhs->type == kTfLiteInt8 && rhs->type == kTfLiteInt8 && !params->adj_x &&
              rank >= 2 && rs.DimensionsCount() == rank && data->quantization != nullptr;
  int batches = 1;
  if (fast) {
    for (int d = 0; d < rank - 2; ++d) {
      if (ls.Dims(d) != rs.Dims(d)) fast = false;
      batches *= ls.Dims(d);
    }
  }
  const int M = fast ? ls.Dims(rank - 2) : 0;
  const int K = fast ? ls.Dims(rank - 1) : 0;
  const int N = fast ? (params->adj_y ? rs.Dims(rank - 2) : rs.Dims(rank - 1)) : 0;
  const int Kr = fast ? (params->adj_y ? rs.Dims(rank - 1) : rs.Dims(rank - 2)) : 0;
  if (!fast || K != Kr || M > kMaxDim || N > kMaxDim ||
      (!params->adj_y && data->rhs_transposed_tensor == nullptr)) {
    return BaseRegistration().invoke(context, node);
  }

  const auto& q = *data->quantization;
  const int32_t za = q.lhs_zero_point, zb = q.rhs_zero_point;
  const int32_t out_zp = q.output_zero_point;
  const int32_t mult = q.output_multiplier;
  const int shift = q.output_shift;
  const int32_t act_min = q.output_activation_min, act_max = q.output_activation_max;
  const int32_t kzz = K * za * zb;

  const int8_t* lhs_data = tflite::micro::GetTensorData<int8_t>(lhs);
  const int8_t* rhs_data = tflite::micro::GetTensorData<int8_t>(rhs);
  int8_t* out_data = tflite::micro::GetTensorData<int8_t>(output);
  int8_t* rhs_t = params->adj_y ? nullptr
                                : tflite::micro::GetTensorData<int8_t>(data->rhs_transposed_tensor);

  int32_t row_term[kMaxDim];  // K·za·zb − zb·Σ_k a[i][k]
  int32_t col_term[kMaxDim];  // −za·Σ_k b[j][k]

  for (int bt = 0; bt < batches; ++bt) {
    const int8_t* A = lhs_data + bt * M * K;
    const int8_t* B;
    if (params->adj_y) {
      B = rhs_data + bt * N * K;  // 已經是 [N][K]
    } else {  // [K][N] 轉置成 [N][K]
      const int8_t* src = rhs_data + bt * K * N;
      for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) rhs_t[n * K + k] = src[k * N + n];
      B = rhs_t;
    }
    for (int i = 0; i < M; ++i) {
      int32_t s = 0;
      for (int k = 0; k < K; ++k) s += A[i * K + k];
      row_term[i] = kzz - zb * s;
    }
    for (int j = 0; j < N; ++j) {
      int32_t s = 0;
      for (int k = 0; k < K; ++k) s += B[j * K + k];
      col_term[j] = -za * s;
    }
    int8_t* Y = out_data + bt * M * N;
    b1::GemmInt8(A, B, M, N, K, [&](int i, int j, int32_t acc) {
      int32_t v = b1::MulByQuantMult(acc + row_term[i] + col_term[j],
                                                               mult, shift);
      v += out_zp;
      if (v < act_min) v = act_min;
      if (v > act_max) v = act_max;
      Y[i * N + j] = static_cast<int8_t>(v);
    });
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_BATCH_MATMUL_B1() {
  return tflite::micro::RegisterOp(BmmB1Init, BmmB1Prepare, BmmB1Eval);
}
