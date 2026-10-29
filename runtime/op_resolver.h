// op_resolver.h — DeiT-Tiny 用到的運算（RISC-V 端 main.cc 與電腦端 tools/gen_offline_params.cc 共用）
// 只註冊用到的運算（B1 的「只加入用到的運算子」）；GELU 是 LiteRT Micro 缺少、自己補上的 kernel。
#ifndef RUNTIME_OP_RESOLVER_H_
#define RUNTIME_OP_RESOLVER_H_

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

TfLiteStatus ParseGeluInt8(const tflite::Operator*, tflite::micro::BuiltinDataAllocator*, void**);
TFLMRegistration Register_GELU_INT8();

// B1（軟體優化）版本的 kernel：編譯時加 -DDEIT_B1（make VARIANT=b1）才會使用，
// 否則用 LiteRT Micro 原本的 reference kernel（B0）。
TFLMRegistration Register_FULLY_CONNECTED_B1();
TFLMRegistration Register_BATCH_MATMUL_B1();
TFLMRegistration Register_CONV_2D_B1();
TFLMRegistration Register_SOFTMAX_B1();
TFLMRegistration Register_ADD_B1();
TFLMRegistration Register_SUB_B1();
TFLMRegistration Register_MUL_B1();
TFLMRegistration Register_MEAN_B1();

using DeitOpResolver = tflite::MicroMutableOpResolver<20>;

inline void RegisterDeitOps(DeitOpResolver& r) {
#ifdef DEIT_B1
  r.AddFullyConnected(Register_FULLY_CONNECTED_B1());
#else
  r.AddFullyConnected();
#endif
#ifdef DEIT_B1
  r.AddBatchMatMul(Register_BATCH_MATMUL_B1());
#else
  r.AddBatchMatMul();
#endif
#ifdef DEIT_B1
  r.AddSoftmax(Register_SOFTMAX_B1());
#else
  r.AddSoftmax();
#endif
#ifdef DEIT_B1
  r.AddMean(Register_MEAN_B1());
#else
  r.AddMean();
#endif
#ifdef DEIT_B1
  r.AddSub(Register_SUB_B1());
#else
  r.AddSub();
#endif
#ifdef DEIT_B1
  r.AddMul(Register_MUL_B1());
#else
  r.AddMul();
#endif
  r.AddRsqrt();
#ifdef DEIT_B1
  r.AddAdd(Register_ADD_B1());
#else
  r.AddAdd();
#endif
  r.AddReshape();
  r.AddTranspose();
  r.AddSlice();
  r.AddStridedSlice();
  r.AddConcatenation();
  r.AddBroadcastTo();
#ifdef DEIT_B1
  r.AddConv2D(Register_CONV_2D_B1());
#else
  r.AddConv2D();
#endif
  r.AddQuantize();
  r.AddGelu(Register_GELU_INT8(), ParseGeluInt8);
}

#endif  // RUNTIME_OP_RESOLVER_H_
