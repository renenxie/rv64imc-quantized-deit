// gelu_int8.cc — 補上 LiteRT Micro 缺少的 GELU 運算（INT8）
//
// LiteRT Micro 沒有 GELU kernel，而 DeiT 的每個 MLP 都會用到。
// 做法與 TFLite（電腦版 LiteRT）的 INT8 GELU 相同：
//   Prepare（初始化，只做一次）：對 256 種 INT8 輸入，用浮點算出 GELU 再量化，建成查表
//   Eval（推論）：out[i] = lut[in[i]]，純整數查表，不用任何浮點
// 建表使用 LiteRT Micro 內建、與 TFLite 相同的 LUTPopulateInt8()。
// 重播模式（TFLM_OFFLINE_PARAMS == 2）：查表直接從離線參數表讀出，目標端完全不做浮點。
#include <cmath>
#include <cstdint>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/flatbuffer_conversions.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/offline_params.h"

namespace {

// 與 <cmath> 的 M_SQRT1_2、M_2_SQRTPI 相同（嚴格 C++ 模式下 newlib 不提供這兩個巨集）
constexpr double kSqrt1_2 = 0.70710678118654752440;
constexpr double k2_SqrtPi = 1.12837916709551257390;

struct GeluOpData {
  int8_t lut[256];
};

// 與 TFLite 的 GeluTransform / GeluTransformApproximate 相同
float GeluTransform(float x) {
  return 0.5f * x * std::erfc(x * static_cast<float>(-kSqrt1_2));
}

float GeluTransformApproximate(float x) {
  return 0.5f * x *
         (1.f + std::tanh(static_cast<float>(k2_SqrtPi * kSqrt1_2) *
                          (x + 0.044715f * x * x * x)));
}

void* GeluInit(TfLiteContext* context, const char*, size_t) {
  return context->AllocatePersistentBuffer(context, sizeof(GeluOpData));
}

TfLiteStatus GeluPrepare(TfLiteContext* context, TfLiteNode* node) {
  tflite::MicroContext* mc = tflite::GetMicroContext(context);
  TfLiteTensor* input = mc->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = mc->AllocateTempOutputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr && output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt8);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);

  auto* data = static_cast<GeluOpData*>(node->user_data);
#if TFLM_OFFLINE_PARAMS == 2
  TF_LITE_ENSURE_OK(context, tflite::micro::offline::Load(
                                 context, tflite::micro::offline::kSlotLut,
                                 data->lut, sizeof(data->lut)));
#else
  const auto* params = static_cast<const TfLiteGeluParams*>(node->builtin_data);
  const bool approximate = params != nullptr && params->approximate;
  if (approximate) {
    tflite::LUTPopulate<int8_t>(input->params.scale, input->params.zero_point,
                                output->params.scale, output->params.zero_point,
                                GeluTransformApproximate, data->lut);
  } else {
    tflite::LUTPopulate<int8_t>(input->params.scale, input->params.zero_point,
                                output->params.scale, output->params.zero_point,
                                GeluTransform, data->lut);
  }
  tflite::micro::offline::Save(tflite::micro::offline::kSlotLut, data->lut,
                               sizeof(data->lut));
#endif
  mc->DeallocateTempTfLiteTensor(input);
  mc->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus GeluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  const auto* data = static_cast<const GeluOpData*>(node->user_data);
  const int n = tflite::micro::GetTensorShape(input).FlatSize();
  const int8_t* in = tflite::micro::GetTensorData<int8_t>(input);
  int8_t* out = tflite::micro::GetTensorData<int8_t>(output);
  for (int i = 0; i < n; ++i) {
    out[i] = data->lut[static_cast<uint8_t>(in[i])];
  }
  return kTfLiteOk;
}

}  // namespace

// 從模型讀出 GELU 的參數（approximate 與否）
TfLiteStatus ParseGeluInt8(const tflite::Operator* op,
                           tflite::micro::BuiltinDataAllocator* allocator,
                           void** builtin_data) {
  auto* params = allocator->AllocatePOD<TfLiteGeluParams>();
  if (params == nullptr) return kTfLiteError;
  params->approximate = false;
  if (const auto* opts = op->builtin_options_as_GeluOptions()) {
    params->approximate = opts->approximate();
  }
  *builtin_data = params;
  return kTfLiteOk;
}

TFLMRegistration Register_GELU_INT8() {
  return tflite::micro::RegisterOp(GeluInit, GeluPrepare, GeluEval);
}
