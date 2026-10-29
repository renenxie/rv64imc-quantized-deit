// requant_b1.h — B1：可內嵌（inline）的 requant
//
// LiteRT Micro 的 MultiplyByQuantizedMultiplier（兩次四捨五入版）寫在 common.cc，不能內嵌，
// 每個元素都要付出函式呼叫的成本。這裡用「同一組」內嵌函式（SaturatingRoundingDoublingHighMul、
// RoundingDivideByPOT，皆來自 LiteRT Micro 的 fixedpoint.h）照原本的步驟重寫，
// 結果逐位元相同，但編譯器可以把它內嵌進迴圈。
#ifndef RUNTIME_KERNELS_REQUANT_B1_H_
#define RUNTIME_KERNELS_REQUANT_B1_H_

#include <cstdint>

#include "tensorflow/lite/micro/kernels/internal/common.h"

namespace b1 {

// 與 tflite::micro::MultiplyByQuantizedMultiplier(int32_t, int32_t, int)（非 TFLITE_SINGLE_ROUNDING）相同
inline int32_t MulByQuantMult(int32_t x, int32_t quantized_multiplier, int shift) {
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  return tflite::micro::RoundingDivideByPOT(
      tflite::micro::SaturatingRoundingDoublingHighMul(x * (1 << left_shift),
                                                       quantized_multiplier),
      right_shift);
}

}  // namespace b1

#endif  // RUNTIME_KERNELS_REQUANT_B1_H_
