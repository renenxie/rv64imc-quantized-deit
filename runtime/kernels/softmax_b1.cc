// softmax_b1.cc — B1：軟體優化的 INT8 SOFTMAX（只用 RV64IMC 原有指令）
//
// B0（reference kernel）對每個元素都要算兩次定點數 exp（加總分母一次、算輸出一次），
// 每次 exp_on_negative_values 約 89 條指令，SOFTMAX 每個輸出約 443 條。
//
// 觀察：輸入是 INT8，所以 input − max_in_row 只可能是 0, −1, …, −255 這 256 種值。
// B1 在初始化時對這 256 種差值各算一次（用與 B0 完全相同的整數函式），存成兩張表：
//   exp_lut[d]：exp_on_negative_values(...) 的結果（Q0.31 定點數）
//   acc_lut[d]：Rescale<12>(exp) 的結果（加總分母用）
// 推論時直接查表；其餘計算（找最大值、倒數、輸出縮放、截斷）與 B0 完全相同，
// 因此結果逐位元相同。差值小於 diff_min 的元素和 B0 一樣不參與加總、輸出 −128。
#include <cstdint>
#include <limits>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/fixedpoint.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/softmax.h"

namespace tflite {
namespace micro {
namespace {

constexpr int kScaledDiffIntegerBits = 5;
constexpr int kAccumulationIntegerBits = 12;
using FixedPointScaledDiff = FixedPoint<int32_t, kScaledDiffIntegerBits>;
using FixedPointAccum = FixedPoint<int32_t, kAccumulationIntegerBits>;
using FixedPoint0 = FixedPoint<int32_t, 0>;

// 第一個成員必須是 SoftmaxParams：LiteRT Micro 原本的 Prepare 會把 user_data 當成它來用
struct OpDataSoftmaxB1 {
  SoftmaxParams base;
  int32_t exp_lut[256];  // d = max − x（0..255）
  int32_t acc_lut[256];
  int num_valid;         // d < num_valid 的元素才參與（對應 input_diff >= diff_min）
  bool fast;
};

const TFLMRegistration& BaseRegistration() {
  static const TFLMRegistration reg = Register_SOFTMAX();
  return reg;
}

void* SoftmaxB1Init(TfLiteContext* context, const char*, size_t) {
  return context->AllocatePersistentBuffer(context, sizeof(OpDataSoftmaxB1));
}

TfLiteStatus SoftmaxB1Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, BaseRegistration().prepare(context, node));
  auto* data = static_cast<OpDataSoftmaxB1*>(node->user_data);
  MicroContext* mc = GetMicroContext(context);
  TfLiteTensor* input = mc->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = mc->AllocateTempOutputTensor(node, 0);
  data->fast = input->type == kTfLiteInt8 && output->type == kTfLiteInt8;
  if (data->fast) {
    const int32_t mult = data->base.input_multiplier;
    const int32_t shift = data->base.input_left_shift;
    const int diff_min = data->base.diff_min;
    data->num_valid = 0;
    for (int d = 0; d < 256; ++d) {
      const int32_t input_diff = -d;
      if (input_diff >= diff_min) {  // 與 B0 相同的計算，只是提前做
        const int32_t rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(input_diff, mult, shift);
        const FixedPoint0 e = exp_on_negative_values(FixedPointScaledDiff::FromRaw(rescaled));
        data->exp_lut[d] = e.raw();
        data->acc_lut[d] = Rescale<kAccumulationIntegerBits>(e).raw();
        data->num_valid = d + 1;
      } else {
        data->exp_lut[d] = 0;
        data->acc_lut[d] = 0;
      }
    }
  }
  mc->DeallocateTempTfLiteTensor(input);
  mc->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus SoftmaxB1Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* data = static_cast<const OpDataSoftmaxB1*>(node->user_data);
  if (!data->fast) return BaseRegistration().invoke(context, node);

  const TfLiteEvalTensor* input = GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = GetEvalOutput(context, node, 0);
  const RuntimeShape shape = GetTensorShape(input);
  const int trailing = shape.DimensionsCount() - 1;
  const int depth = shape.Dims(trailing);
  const int outer = shape.FlatSize() / depth;
  const int8_t* x = GetTensorData<int8_t>(input);
  int8_t* y = GetTensorData<int8_t>(output);
  const int32_t* exp_lut = data->exp_lut;
  const int32_t* acc_lut = data->acc_lut;
  const int num_valid = data->num_valid;

  for (int i = 0; i < outer; ++i) {
    const int8_t* xr = x + i * depth;
    int8_t* yr = y + i * depth;
    int32_t max_in_row = std::numeric_limits<int8_t>::min();
    for (int c = 0; c < depth; ++c) {
      if (xr[c] > max_in_row) max_in_row = xr[c];
    }
    // 分母：與 B0 相同的定點數加總（FixedPoint 的加法就是 int32 加法）
    int32_t sum_raw = 0;
    for (int c = 0; c < depth; ++c) sum_raw += acc_lut[max_in_row - xr[c]];

    int num_bits_over_unit;
    const FixedPoint0 shifted_scale = FixedPoint0::FromRaw(
        GetReciprocal(sum_raw, kAccumulationIntegerBits, &num_bits_over_unit));
    const int exponent = num_bits_over_unit + 31 - 8;

    for (int c = 0; c < depth; ++c) {
      const int d = max_in_row - xr[c];
      if (d < num_valid) {
        const FixedPoint0 e = FixedPoint0::FromRaw(exp_lut[d]);
        int32_t v = RoundingDivideByPOT((shifted_scale * e).raw(), exponent) - 128;
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        yr[c] = static_cast<int8_t>(v);
      } else {
        yr[c] = -128;
      }
    }
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace micro
}  // namespace tflite

TFLMRegistration Register_SOFTMAX_B1() {
  return tflite::micro::RegisterOp(tflite::micro::SoftmaxB1Init, tflite::micro::SoftmaxB1Prepare,
                                   tflite::micro::SoftmaxB1Eval);
}
