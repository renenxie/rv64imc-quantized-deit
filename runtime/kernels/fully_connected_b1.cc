// fully_connected_b1.cc — B1：軟體優化的 INT8 FULLY_CONNECTED（只用 RV64IMC 原有指令）
//
// 與 LiteRT Micro 的 reference kernel（B0）算出「逐位元相同」的結果，只改寫法：
//
//  1. zero-point 移出內層迴圈
//       B0：acc = Σ_k w[n][k] · (x[k] + input_offset) + bias[n]     每個 MAC 多一個加法
//       B1：acc = Σ_k w[n][k] · x[k]  +  (bias[n] + input_offset · Σ_k w[n][k])
//                                         └──────── 初始化時算好（整數）────────┘
//  2. 暫存器分塊（register blocking）：一次算 2 個 token × 4 個輸出通道 = 8 個累加器，
//     每載入 1 個輸入給 4 個通道用、每載入 1 個權重給 2 個 token 用 → load / MAC 從 2 降到 0.75
//  3. 內層迴圈展開 4 次：指標遞增與迴圈判斷分攤到 4 個 k
//
// int32 累加：|x|,|w| ≤ 128，K ≤ 768 → |Σ| ≤ 128·128·768 ≈ 1.26e7，遠小於 2^31，不會溢位，
// 所以改變加法順序不影響結果。requant 與 B0（已改成單次四捨五入的 per-channel 版本）完全相同。
//
// 只處理 DeiT 用到的情況（INT8 輸入 / 輸出、per-channel INT8 權重、INT32 bias、無 fused activation
// 以外的上下限照常截斷）；其他情況交回 LiteRT Micro 原本的 kernel。
#include <cstdint>

#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

// 關掉「暫存器配置前」的指令排程：它會把很多 load 提前，暫存器不夠而溢出到 stack，
// 內層迴圈每圈多出約 20 條 ld/sd（實測 FC 3.95G → 3.25G 條指令）。只影響這個檔案。
#pragma GCC optimize("no-schedule-insns")

namespace {

using tflite::micro::OpDataFullyConnected;

// 第一個成員必須是 OpDataFullyConnected：LiteRT Micro 原本的 Prepare 會把 user_data 當成它來用
struct OpDataFcB1 {
  OpDataFullyConnected base;
  int32_t* bias_eff;  // bias[n] + input_offset · Σ_k w[n][k]（每個輸出通道一個）
  bool fast;          // 是否走 B1 的快速路徑
};

const TFLMRegistration& BaseRegistration() {
  static const TFLMRegistration reg = tflite::micro::Register_FULLY_CONNECTED();
  return reg;
}

void* FcB1Init(TfLiteContext* context, const char*, size_t) {
  return context->AllocatePersistentBuffer(context, sizeof(OpDataFcB1));
}

TfLiteStatus FcB1Prepare(TfLiteContext* context, TfLiteNode* node) {
  // 先跑 LiteRT Micro 原本的 Prepare：檢查形狀、取得 per-channel multiplier/shift（離線參數表）
  TF_LITE_ENSURE_OK(context, BaseRegistration().prepare(context, node));
  auto* data = static_cast<OpDataFcB1*>(node->user_data);
  data->fast = false;
  data->bias_eff = nullptr;

  tflite::MicroContext* mc = tflite::GetMicroContext(context);
  TfLiteTensor* input = mc->AllocateTempInputTensor(node, 0);
  TfLiteTensor* filter = mc->AllocateTempInputTensor(node, 1);
  TfLiteTensor* bias = mc->AllocateTempInputTensor(node, 2);
  TfLiteTensor* output = mc->AllocateTempOutputTensor(node, 0);

  if (input->type == kTfLiteInt8 && filter->type == kTfLiteInt8 &&
      output->type == kTfLiteInt8 && data->base.is_per_channel &&
      (bias == nullptr || bias->type == kTfLiteInt32) && filter->dims->size == 2) {
    const int n_out = filter->dims->data[0];
    const int k_in = filter->dims->data[1];
    data->bias_eff = static_cast<int32_t*>(
        context->AllocatePersistentBuffer(context, n_out * sizeof(int32_t)));
    const int8_t* w = filter->data.int8;
    const int32_t* b = bias ? bias->data.i32 : nullptr;
    const int32_t input_offset = -data->base.input_zero_point;
    for (int n = 0; n < n_out; ++n) {
      int32_t wsum = 0;
      for (int k = 0; k < k_in; ++k) wsum += w[n * k_in + k];
      data->bias_eff[n] = (b ? b[n] : 0) + input_offset * wsum;
    }
    data->fast = true;
  }

  mc->DeallocateTempTfLiteTensor(input);
  mc->DeallocateTempTfLiteTensor(filter);
  if (bias) mc->DeallocateTempTfLiteTensor(bias);
  mc->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

// 與 B0 的 per-channel requant 完全相同（單次四捨五入）
inline int8_t Requant(int32_t acc, int32_t mult, int32_t shift, int32_t out_off,
                      int32_t act_min, int32_t act_max) {
  const int64_t total_shift = 31 - shift;
  const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
  int32_t v = static_cast<int32_t>((static_cast<int64_t>(acc) * mult + round) >> total_shift);
  v += out_off;
  if (v < act_min) v = act_min;
  if (v > act_max) v = act_max;
  return static_cast<int8_t>(v);
}

TfLiteStatus FcB1Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* data = static_cast<const OpDataFcB1*>(node->user_data);
  if (!data->fast) return BaseRegistration().invoke(context, node);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* filter = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  const tflite::RuntimeShape out_shape = tflite::micro::GetTensorShape(output);
  const int out_dims = out_shape.DimensionsCount();
  const int batches = tflite::FlatSizeSkipDim(out_shape, out_dims - 1);
  const int n_out = out_shape.Dims(out_dims - 1);
  const int k_in = tflite::micro::GetTensorShape(filter).Dims(1);

  const int8_t* __restrict x = tflite::micro::GetTensorData<int8_t>(input);
  const int8_t* __restrict w = tflite::micro::GetTensorData<int8_t>(filter);
  int8_t* __restrict y = tflite::micro::GetTensorData<int8_t>(output);
  const int32_t* bias_eff = data->bias_eff;
  const int32_t* mult = data->base.per_channel_output_multiplier;
  const int32_t* shift = data->base.per_channel_output_shift;
  const int32_t out_off = data->base.output_zero_point;
  const int32_t act_min = data->base.output_activation_min;
  const int32_t act_max = data->base.output_activation_max;

  int b = 0;
  // 主體：2 個 token × 4 個輸出通道
  for (; b + 1 < batches; b += 2) {
    const int8_t* x0 = x + b * k_in;
    const int8_t* x1 = x0 + k_in;
    int8_t* y0 = y + b * n_out;
    int8_t* y1 = y0 + n_out;
    int n = 0;
    for (; n + 3 < n_out; n += 4) {
      const int8_t* w0 = w + n * k_in;
      const int8_t* w1 = w0 + k_in;
      const int8_t* w2 = w1 + k_in;
      const int8_t* w3 = w2 + k_in;
      int32_t a00 = 0, a01 = 0, a02 = 0, a03 = 0;
      int32_t a10 = 0, a11 = 0, a12 = 0, a13 = 0;
      // 用指標 + 固定位移存取（lb rd, o(ptr)），每 4 個 k 才遞增一次指標
      const int8_t* p0 = x0;
      const int8_t* p1 = x1;
      const int8_t* q0 = w0;
      const int8_t* q1 = w1;
      const int8_t* q2 = w2;
      const int8_t* q3 = w3;
      const int8_t* const p_end4 = x0 + (k_in & ~3);
#define FC_B1_STEP(o)                                                         \
  {                                                                           \
    const int32_t i0 = p0[o], i1 = p1[o];                                     \
    const int32_t c0 = q0[o], c1 = q1[o], c2 = q2[o], c3 = q3[o];             \
    a00 += i0 * c0; a01 += i0 * c1; a02 += i0 * c2; a03 += i0 * c3;           \
    a10 += i1 * c0; a11 += i1 * c1; a12 += i1 * c2; a13 += i1 * c3;           \
  }
      while (p0 != p_end4) {
        FC_B1_STEP(0) FC_B1_STEP(1) FC_B1_STEP(2) FC_B1_STEP(3)
        p0 += 4; p1 += 4; q0 += 4; q1 += 4; q2 += 4; q3 += 4;
      }
      while (p0 != x0 + k_in) {  // K 不是 4 的倍數時的尾端
        FC_B1_STEP(0)
        ++p0; ++p1; ++q0; ++q1; ++q2; ++q3;
      }
#undef FC_B1_STEP
      const int32_t acc0[4] = {a00, a01, a02, a03};
      const int32_t acc1[4] = {a10, a11, a12, a13};
      for (int j = 0; j < 4; ++j) {
        const int c = n + j;
        y0[c] = Requant(acc0[j] + bias_eff[c], mult[c], shift[c], out_off, act_min, act_max);
        y1[c] = Requant(acc1[j] + bias_eff[c], mult[c], shift[c], out_off, act_min, act_max);
      }
    }
    for (; n < n_out; ++n) {  // 剩下不足 4 個的輸出通道
      const int8_t* wn = w + n * k_in;
      int32_t a0 = 0, a1 = 0;
      for (int k = 0; k < k_in; ++k) {
        a0 += x0[k] * wn[k];
        a1 += x1[k] * wn[k];
      }
      y0[n] = Requant(a0 + bias_eff[n], mult[n], shift[n], out_off, act_min, act_max);
      y1[n] = Requant(a1 + bias_eff[n], mult[n], shift[n], out_off, act_min, act_max);
    }
  }
  // 剩下的單一 token（例如最後分類層只有 1 個 token）：1 個 token × 4 個輸出通道
  for (; b < batches; ++b) {
    const int8_t* x0 = x + b * k_in;
    int8_t* y0 = y + b * n_out;
    int n = 0;
    for (; n + 3 < n_out; n += 4) {
      const int8_t* w0 = w + n * k_in;
      const int8_t* w1 = w0 + k_in;
      const int8_t* w2 = w1 + k_in;
      const int8_t* w3 = w2 + k_in;
      int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#pragma GCC unroll 4
      for (int k = 0; k < k_in; ++k) {
        const int32_t i0 = x0[k];
        a0 += i0 * w0[k]; a1 += i0 * w1[k]; a2 += i0 * w2[k]; a3 += i0 * w3[k];
      }
      const int32_t acc[4] = {a0, a1, a2, a3};
      for (int j = 0; j < 4; ++j) {
        const int c = n + j;
        y0[c] = Requant(acc[j] + bias_eff[c], mult[c], shift[c], out_off, act_min, act_max);
      }
    }
    for (; n < n_out; ++n) {
      const int8_t* wn = w + n * k_in;
      int32_t a0 = 0;
      for (int k = 0; k < k_in; ++k) a0 += x0[k] * wn[k];
      y0[n] = Requant(a0 + bias_eff[n], mult[n], shift[n], out_off, act_min, act_max);
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_FULLY_CONNECTED_B1() {
  return tflite::micro::RegisterOp(FcB1Init, FcB1Prepare, FcB1Eval);
}
