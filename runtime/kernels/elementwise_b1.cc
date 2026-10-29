// elementwise_b1.cc — B1：軟體優化的 INT8 ADD / SUB / MUL / MEAN（只用 RV64IMC 原有指令）
//
// 這些運算組成 LayerNorm（MEAN → SUB → MUL → MEAN → ADD → RSQRT → MUL → MUL → ADD）與 residual。
// B0 的 reference kernel 用通用的 4D/6D broadcast 迴圈，每個元素都要算多維索引，
// ADD 每個輸出約 117 條指令、SUB 105 條、MUL 59 條、MEAN 每個輸出約 25,800 條。
//
// B1 做法（結果與 B0 逐位元相同）：
//  1. broadcast 只處理 DeiT 出現的形式：兩邊形狀相同、純量、每行一個值 [R,1]、每列一個值 [1,C]。
//     輸出看成 R 列 × C 行，每個輸入用「列步長、行步長」走訪，不必算多維索引。
//  2. ADD / SUB：B0 每個元素對兩個輸入各做一次 requant（輸入 → 共同尺度），但輸入只有 256 種值，
//     所以初始化時把這兩個 requant 各做成 256 格查表；每個元素只剩一次輸出 requant。
//  3. MUL：公式不變，只省掉索引計算。
//  4. MEAN：只沿最後一維平均時，直接逐列加總；最後的 requant 與 B0 相同。
// 其他形狀交回 LiteRT Micro 原本的 kernel。
#include <cstdint>

#include "kernels/requant_b1.h"

#include "tensorflow/lite/micro/kernels/add.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/mul.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/kernels/sub.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

namespace tflite {
namespace micro {
namespace {

// ---------------------------------------------------------------- broadcast 形式
// 輸出攤平成 R × C（C = 最後一維）。輸入 i 的元素位置 = r·rs + c·cs。
struct Walk {
  int rs, cs;
};

bool MakeWalk(const TfLiteEvalTensor* in, const TfLiteEvalTensor* out, Walk* w) {
  const int on = out->dims->size, in_n = in->dims->size;
  if (on < 1 || in_n > on) return false;
  const int C = out->dims->data[on - 1];
  auto in_dim = [&](int d) {  // 對齊到輸出的維度（前面補 1）
    const int k = d - (on - in_n);
    return k < 0 ? 1 : in->dims->data[k];
  };
  bool same = true, all_one = true, last_only = true, row_only = true;
  for (int d = 0; d < on; ++d) {
    const int a = in_dim(d), o = out->dims->data[d];
    if (a != o) same = false;
    if (a != 1) all_one = false;
    if (d < on - 1) {
      if (a != 1) last_only = false;
      if (a != o) row_only = false;
    } else {
      if (a != C) last_only = false;
      if (a != 1) row_only = false;
    }
  }
  if (same) *w = {C, 1};
  else if (all_one) *w = {0, 0};
  else if (last_only) *w = {0, 1};  // [1, C]：每列共用同一組
  else if (row_only) *w = {1, 0};   // [R, 1]：每行一個值
  else return false;
  return true;
}

template <typename F>
inline void BinaryLoop(const int8_t* x, Walk wx, const int8_t* y, Walk wy, int8_t* out, int R,
                       int C, F f) {
  for (int r = 0; r < R; ++r) {
    const int8_t* px = x + r * wx.rs;
    const int8_t* py = y + r * wy.rs;
    int8_t* po = out + r * C;
    for (int c = 0; c < C; ++c) {
      po[c] = f(*px, *py);
      px += wx.cs;
      py += wy.cs;
    }
  }
}

inline int8_t Clamp8(int32_t v, int32_t lo, int32_t hi) {
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return static_cast<int8_t>(v);
}

// ---------------------------------------------------------------- ADD / SUB
// 兩者的 OpData 欄位相同（OpDataAdd / OpDataSub），用樣板共用
template <typename OpData>
struct AddSubB1 {
  OpData base;   // 第一個成員：原本的 Prepare 會把 user_data 當成它
  int32_t lut1[256];  // MultiplyByQuantizedMultiplierSmallerThanOneExp((x + off1) << left_shift, m1, s1)
  int32_t lut2[256];
  bool fast;
};

template <typename OpData>
void FillAddSubLuts(AddSubB1<OpData>* d) {
  const OpData& p = d->base;
  for (int v = -128; v < 128; ++v) {  // 與 B0 每個元素做的計算完全相同，只是提前做
    d->lut1[v + 128] = MultiplyByQuantizedMultiplierSmallerThanOneExp(
        (p.input1_offset + v) * (1 << p.left_shift), p.input1_multiplier, p.input1_shift);
    d->lut2[v + 128] = MultiplyByQuantizedMultiplierSmallerThanOneExp(
        (p.input2_offset + v) * (1 << p.left_shift), p.input2_multiplier, p.input2_shift);
  }
}

const TFLMRegistration& AddBase() {
  static const TFLMRegistration reg = Register_ADD();
  return reg;
}
const TFLMRegistration& SubBase() {
  static const TFLMRegistration reg = Register_SUB();
  return reg;
}

template <typename OpData, bool kIsSub>
void* AddSubInit(TfLiteContext* context, const char*, size_t) {
  return context->AllocatePersistentBuffer(context, sizeof(AddSubB1<OpData>));
}

template <typename OpData, bool kIsSub>
TfLiteStatus AddSubPrepare(TfLiteContext* context, TfLiteNode* node) {
  const TFLMRegistration& base = kIsSub ? SubBase() : AddBase();
  TF_LITE_ENSURE_OK(context, base.prepare(context, node));
  auto* d = static_cast<AddSubB1<OpData>*>(node->user_data);
  MicroContext* mc = GetMicroContext(context);
  TfLiteTensor* in1 = mc->AllocateTempInputTensor(node, 0);
  TfLiteTensor* in2 = mc->AllocateTempInputTensor(node, 1);
  TfLiteTensor* out = mc->AllocateTempOutputTensor(node, 0);
  d->fast = in1->type == kTfLiteInt8 && in2->type == kTfLiteInt8 && out->type == kTfLiteInt8;
  if (d->fast) FillAddSubLuts(d);
  mc->DeallocateTempTfLiteTensor(in1);
  mc->DeallocateTempTfLiteTensor(in2);
  mc->DeallocateTempTfLiteTensor(out);
  return kTfLiteOk;
}

template <typename OpData, bool kIsSub>
TfLiteStatus AddSubEval(TfLiteContext* context, TfLiteNode* node) {
  const auto* d = static_cast<const AddSubB1<OpData>*>(node->user_data);
  const TFLMRegistration& base = kIsSub ? SubBase() : AddBase();
  const TfLiteEvalTensor* in1 = GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* in2 = GetEvalInput(context, node, 1);
  TfLiteEvalTensor* out = GetEvalOutput(context, node, 0);
  Walk w1, w2;
  if (!d->fast || !MakeWalk(in1, out, &w1) || !MakeWalk(in2, out, &w2)) {
    return base.invoke(context, node);
  }
  const int C = out->dims->data[out->dims->size - 1];
  const int R = ElementCount(*out->dims) / C;
  const OpData& p = d->base;
  const int32_t* l1 = d->lut1 + 128;
  const int32_t* l2 = d->lut2 + 128;
  const int32_t om = p.output_multiplier, os = p.output_shift, oo = p.output_offset;
  const int32_t lo = p.output_activation_min, hi = p.output_activation_max;
  BinaryLoop(GetTensorData<int8_t>(in1), w1, GetTensorData<int8_t>(in2), w2,
             GetTensorData<int8_t>(out), R, C, [&](int8_t x, int8_t y) {
               const int32_t raw = kIsSub ? l1[x] - l2[y] : l1[x] + l2[y];
               return Clamp8(MultiplyByQuantizedMultiplierSmallerThanOneExp(raw, om, os) + oo,
                             lo, hi);
             });
  return kTfLiteOk;
}

// ---------------------------------------------------------------- MUL
const TFLMRegistration& MulBase() {
  static const TFLMRegistration reg = Register_MUL();
  return reg;
}

TfLiteStatus MulB1Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* p = static_cast<const OpDataMul*>(node->user_data);
  const TfLiteEvalTensor* in1 = GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* in2 = GetEvalInput(context, node, 1);
  TfLiteEvalTensor* out = GetEvalOutput(context, node, 0);
  Walk w1, w2;
  if (in1->type != kTfLiteInt8 || in2->type != kTfLiteInt8 || out->type != kTfLiteInt8 ||
      !MakeWalk(in1, out, &w1) || !MakeWalk(in2, out, &w2)) {
    return MulBase().invoke(context, node);
  }
  const int C = out->dims->data[out->dims->size - 1];
  const int R = ElementCount(*out->dims) / C;
  const int32_t o1 = -p->input1_zero_point, o2 = -p->input2_zero_point;
  const int32_t om = p->output_multiplier, os = p->output_shift, oo = p->output_zero_point;
  const int32_t lo = p->output_activation_min, hi = p->output_activation_max;
  BinaryLoop(GetTensorData<int8_t>(in1), w1, GetTensorData<int8_t>(in2), w2,
             GetTensorData<int8_t>(out), R, C, [&](int8_t x, int8_t y) {
               return Clamp8(oo + b1::MulByQuantMult((o1 + x) * (o2 + y), om, os), lo,
                             hi);
             });
  return kTfLiteOk;
}

// ---------------------------------------------------------------- MEAN（沿最後一維）
const TFLMRegistration& MeanBase() {
  static const TFLMRegistration reg = Register_MEAN();
  return reg;
}

TfLiteStatus MeanB1Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* d = static_cast<const OpDataReduce*>(node->user_data);
  const TfLiteEvalTensor* in = GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = GetEvalInput(context, node, 1);
  TfLiteEvalTensor* out = GetEvalOutput(context, node, 0);
  const int rank = in->dims->size;
  bool fast = in->type == kTfLiteInt8 && out->type == kTfLiteInt8 &&
              ElementCount(*axis->dims) == 1 && rank >= 1;
  if (fast) {
    const int a = GetTensorData<int32_t>(axis)[0];
    fast = (a == rank - 1 || a == -1) && in->dims->data[rank - 1] > 0;
  }
  if (!fast) return MeanBase().invoke(context, node);

  const int C = in->dims->data[rank - 1];
  const int R = ElementCount(*in->dims) / C;
  // 與 B0（reference_ops::QuantizedMeanOrSum）相同：把「除以元素個數」併進 multiplier
  int shift = 63 - CountLeadingZeros(static_cast<uint64_t>(C));
  if (shift > 32) shift = 32;
  if (shift > 31 + d->shift) shift = 31 + d->shift;
  const int32_t mult =
      static_cast<int32_t>((static_cast<int64_t>(d->multiplier) << shift) / C);
  const int out_shift = d->shift - shift;
  const int32_t zp_sum = d->input_zp * C;

  const int8_t* x = GetTensorData<int8_t>(in);
  int8_t* y = GetTensorData<int8_t>(out);
  for (int r = 0; r < R; ++r) {
    const int8_t* xr = x + r * C;
    int32_t s = 0;
    for (int c = 0; c < C; ++c) s += xr[c];
    const int32_t v = b1::MulByQuantMult(s - zp_sum, mult, out_shift) + d->output_zp;
    y[r] = Clamp8(v, -128, 127);
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace micro
}  // namespace tflite

TFLMRegistration Register_ADD_B1() {
  using namespace tflite::micro;
  return RegisterOp(AddSubInit<OpDataAdd, false>, AddSubPrepare<OpDataAdd, false>,
                    AddSubEval<OpDataAdd, false>);
}

TFLMRegistration Register_SUB_B1() {
  using namespace tflite::micro;
  return RegisterOp(AddSubInit<OpDataSub, true>, AddSubPrepare<OpDataSub, true>,
                    AddSubEval<OpDataSub, true>);
}

TFLMRegistration Register_MUL_B1() {
  using namespace tflite::micro;
  TFLMRegistration r = MulBase();
  r.invoke = MulB1Eval;  // Init / Prepare 沿用原本的
  return r;
}

TFLMRegistration Register_MEAN_B1() {
  using namespace tflite::micro;
  TFLMRegistration r = MeanBase();
  r.invoke = MeanB1Eval;
  return r;
}
