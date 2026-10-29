// platform.cc — LiteRT Micro 需要平台提供的函式（取代 tflm 內建的預設版本）
//
//   * GetCurrentTimeTicks()：給 MicroProfiler 量測每個運算用。
//     Spike 不模擬時間，所以用 instret（已執行指令數）當作「tick」，
//     profiler 印出的數字就是每個運算的動態指令數。
//   * C++ 執行期需要的少數符號（不使用 exceptions / RTTI / 動態記憶體）
#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {
namespace micro {

uint32_t ticks_per_second() { return 0; }  // 0 = 不換算成秒，直接看 tick（指令數）

uint32_t GetCurrentTimeTicks() {
  uint64_t x;
  __asm__ volatile("rdinstret %0" : "=r"(x));
  return static_cast<uint32_t>(x);
}

}  // namespace micro
}  // namespace tflite

// 有 virtual destructor 的類別會參考到 operator delete，但本程式從不 delete
void operator delete(void*) noexcept {}
void operator delete(void*, std::size_t) noexcept {}
extern "C" void __cxa_pure_virtual() { while (true) {} }
