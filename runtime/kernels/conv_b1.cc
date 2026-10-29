// conv_b1.cc — B1：軟體優化的 INT8 CONV_2D（只用 RV64IMC 原有指令）
//
// DeiT 只有一個卷積：patch embedding，16×16、stride 16、輸入 3 通道、輸出 192 通道、沒有 padding。
// B0（reference kernel）的最內層迴圈只轉 3 圈（輸入通道數），每轉 3 圈就要重算位置、檢查邊界，
// 開銷比乘加還多（每個 MAC 約 18 條指令）。
//
// B1 做法（im2col）：
//   1. 把每個 16×16×3 的輸入區塊攤平成一列（768 個 INT8），196 個區塊 → 196×768 的矩陣
//      （每列由 16 段、每段 48 個連續 byte 組成，直接複製）
//   2. 權重 [192][16][16][3] 本來就是 [192][768]，順序與攤平後的區塊一致
//   3. 於是卷積 = 196×768 乘 768×192 的矩陣乘法，用與 FULLY_CONNECTED 相同的內積核心（gemm_b1.h）
//   4. zero-point 移出內層迴圈：bias_eff[c] = bias[c] + input_offset · Σ w[c]（初始化時算好）
//
// 與 B0 逐位元相同：同樣的整數乘加、同樣的 requant（tflite::micro::MultiplyByQuantizedMultiplier）。
// 只處理沒有 padding、dilation = 1 的 INT8 per-channel 卷積；其他情況交回原本的 kernel。
#pragma GCC optimize("no-schedule-insns")  // 理由見 gemm_b1.h

#include <cstdint>

#include "kernels/requant_b1.h"
#include <cstring>

#include "kernels/gemm_b1.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace {

using tflite::micro::OpDataConv;

// 第一個成員必須是 OpDataConv：LiteRT Micro 原本的 Prepare 會把 user_data 當成它來用
struct OpDataConvB1 {
  OpDataConv base;
  int32_t* bias_eff;  // bias[c] + input_offset · Σ w[c]
  int scratch_index;  // im2col 矩陣（每個 batch 共用）
  bool fast;
};

const TFLMRegistration& BaseRegistration() {
  static const TFLMRegistration reg = tflite::micro::Register_CONV_2D();
  return reg;
}

void* ConvB1Init(TfLiteContext* context, const char*, size_t) {
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConvB1));
}

TfLiteStatus ConvB1Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, BaseRegistration().prepare(context, node));
  auto* data = static_cast<OpDataConvB1*>(node->user_data);
  const auto& params = *static_cast<const TfLiteConvParams*>(node->builtin_data);
  data->fast = false;

  tflite::MicroContext* mc = tflite::GetMicroContext(context);
  TfLiteTensor* input = mc->AllocateTempInputTensor(node, 0);
  TfLiteTensor* filter = mc->AllocateTempInputTensor(node, 1);
  TfLiteTensor* bias = mc->AllocateTempInputTensor(node, 2);
  TfLiteTensor* output = mc->AllocateTempOutputTensor(node, 0);

  if (input->type == kTfLiteInt8 && filter->type == kTfLiteInt8 && output->type == kTfLiteInt8 &&
      (bias == nullptr || bias->type == kTfLiteInt32) && params.dilation_height_factor == 1 &&
      params.dilation_width_factor == 1 && data->base.padding.height == 0 &&
      data->base.padding.width == 0 && data->base.padding.height_offset == 0 &&
      data->base.padding.width_offset == 0 && filter->dims->size == 4) {
    const int cout = filter->dims->data[0];
    const int k = filter->dims->data[1] * filter->dims->data[2] * filter->dims->data[3];
    const int m = output->dims->data[1] * output->dims->data[2];
    data->bias_eff =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(context, cout * sizeof(int32_t)));
    const int8_t* w = filter->data.int8;
    const int32_t* b = bias ? bias->data.i32 : nullptr;
    const int32_t input_offset = -data->base.input_zero_point;
    for (int c = 0; c < cout; ++c) {
      int32_t wsum = 0;
      for (int i = 0; i < k; ++i) wsum += w[c * k + i];
      data->bias_eff[c] = (b ? b[c] : 0) + input_offset * wsum;
    }
    TF_LITE_ENSURE_OK(context,
                      context->RequestScratchBufferInArena(context, m * k, &data->scratch_index));
    data->fast = true;
  }

  mc->DeallocateTempTfLiteTensor(input);
  mc->DeallocateTempTfLiteTensor(filter);
  if (bias) mc->DeallocateTempTfLiteTensor(bias);
  mc->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus ConvB1Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* data = static_cast<const OpDataConvB1*>(node->user_data);
  if (!data->fast) return BaseRegistration().invoke(context, node);
  const auto& params = *static_cast<const TfLiteConvParams*>(node->builtin_data);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* filter = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  const tflite::RuntimeShape is = tflite::micro::GetTensorShape(input);
  const tflite::RuntimeShape fs = tflite::micro::GetTensorShape(filter);
  const tflite::RuntimeShape os = tflite::micro::GetTensorShape(output);
  const int batches = is.Dims(0), in_h = is.Dims(1), in_w = is.Dims(2), in_c = is.Dims(3);
  const int cout = fs.Dims(0), kh = fs.Dims(1), kw = fs.Dims(2);
  const int out_h = os.Dims(1), out_w = os.Dims(2);
  const int sh = params.stride_height, sw = params.stride_width;
  const int K = kh * kw * in_c;
  const int M = out_h * out_w;
  const int seg = kw * in_c;  // 每段連續 byte 數（16 × 3 = 48）

  const int8_t* x = tflite::micro::GetTensorData<int8_t>(input);
  const int8_t* w = tflite::micro::GetTensorData<int8_t>(filter);
  int8_t* y = tflite::micro::GetTensorData<int8_t>(output);
  int8_t* cols = static_cast<int8_t*>(context->GetScratchBuffer(context, data->scratch_index));
  const int32_t* bias_eff = data->bias_eff;
  const int32_t* mult = data->base.per_channel_output_multiplier;
  const int32_t* shift = data->base.per_channel_output_shift;
  const int32_t out_off = data->base.output_zero_point;
  const int32_t act_min = data->base.output_activation_min;
  const int32_t act_max = data->base.output_activation_max;

  for (int b = 0; b < batches; ++b) {
    // im2col：每個輸出位置一列，由 kh 段、每段 kw·in_c 個連續 byte 組成
    for (int oy = 0; oy < out_h; ++oy) {
      for (int ox = 0; ox < out_w; ++ox) {
        int8_t* row = cols + (oy * out_w + ox) * K;
        for (int ky = 0; ky < kh; ++ky) {
          const int8_t* src = x + ((b * in_h + oy * sh + ky) * in_w + ox * sw) * in_c;
          std::memcpy(row + ky * seg, src, seg);
        }
      }
    }
    int8_t* yb = y + b * M * cout;
    b1::GemmInt8(cols, w, M, cout, K, [&](int i, int j, int32_t acc) {
      int32_t v = b1::MulByQuantMult(acc + bias_eff[j], mult[j], shift[j]);
      v += out_off;
      if (v < act_min) v = act_min;
      if (v > act_max) v = act_max;
      yb[i * cout + j] = static_cast<int8_t>(v);
    });
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D_B1() {
  return tflite::micro::RegisterOp(ConvB1Init, ConvB1Prepare, ConvB1Eval);
}
