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

using DeitOpResolver = tflite::MicroMutableOpResolver<20>;

inline void RegisterDeitOps(DeitOpResolver& r) {
#ifdef DEIT_B1
  r.AddFullyConnected(Register_FULLY_CONNECTED_B1());
#else
  r.AddFullyConnected();
#endif
  r.AddBatchMatMul();
  r.AddSoftmax();
  r.AddMean();
  r.AddSub();
  r.AddMul();
  r.AddRsqrt();
  r.AddAdd();
  r.AddReshape();
  r.AddTranspose();
  r.AddSlice();
  r.AddStridedSlice();
  r.AddConcatenation();
  r.AddBroadcastTo();
  r.AddConv2D();
  r.AddQuantize();
  r.AddGelu(Register_GELU_INT8(), ParseGeluInt8);
}

#endif  // RUNTIME_OP_RESOLVER_H_
