// [期中project 新增] offline_params.h — 把 kernel 初始化（Prepare）時的浮點計算移到離線
//
// LiteRT Micro 在 Prepare 時用浮點把 scale（float）換算成整數參數，例如
// requant 的 multiplier/shift、softmax 的 diff_min、GELU 查表。RV64IMC 沒有浮點硬體，
// 這些計算會變成軟體浮點。本機制讓目標端完全不做浮點：
//
//   TFLM_OFFLINE_PARAMS = 0  關閉：原本行為（Prepare 時用浮點計算）
//                       = 1  記錄：在電腦（x86）上執行原本的計算，把算出的整數參數存成離線表
//                       = 2  重播：目標端（RV64IMC）直接讀離線表，不執行任何浮點計算
//                       = 3  稽核：目標端照原本方式計算，並逐一與離線表比對（測試用）
//
// 每筆參數以「運算編號（op index）+ 欄位編號（slot）」為鍵。運算編號由
// MicroInterpreterGraph 在呼叫每個運算的 Prepare 前設定。
#ifndef TENSORFLOW_LITE_MICRO_OFFLINE_PARAMS_H_
#define TENSORFLOW_LITE_MICRO_OFFLINE_PARAMS_H_

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/micro_common.h"

#ifndef TFLM_OFFLINE_PARAMS
#define TFLM_OFFLINE_PARAMS 0
#endif

namespace tflite {
namespace micro {
namespace offline {

// 欄位編號（同一個運算內不可重複）
enum Slot : int {
  kSlotMultiplier = 0,       // 各 kernel 的主要 multiplier/shift 組
  kSlotPerChannelMult = 1,   // per-channel multiplier 陣列
  kSlotPerChannelShift = 2,  // per-channel shift 陣列
  kSlotLut = 3,              // 查表（GELU）
  kSlotActivation = 4,       // 有 fused activation 時的 int8 上下限
  kSlotScaleBits = 5,        // 直接保存 double scale 的位元（QUANTIZE 用）
  kSlotMultiplier2 = 6,      // 同一運算的第二組 multiplier/shift（MEAN 的 PrepareSimple）
};

// 由 MicroInterpreterGraph 在每個運算 Prepare 前呼叫
void SetCurrentOp(int op_index);
int CurrentOp();

// 記錄模式：存下；稽核模式：與離線表比對；其他模式：不做事
void Save(int slot, const void* data, size_t bytes);

// 重播模式：從離線表讀出；找不到或大小不符時回傳 kTfLiteError
TfLiteStatus Load(TfLiteContext* context, int slot, void* data, size_t bytes);

// 記錄模式：取得整份離線表（電腦端寫成檔案）
const uint8_t* RecordedTable(size_t* bytes);

// 稽核模式：比對了幾筆、幾筆不一致
uint32_t AuditChecked();
uint32_t AuditMismatches();

// 以位元比較兩個 float 是否相同（不使用浮點運算；用於 scale 是否相等的檢查）
inline bool SameFloatBits(float a, float b) {
  union { float f; uint32_t u; } x{a}, y{b};
  return x.u == y.u;
}

}  // namespace offline
}  // namespace micro
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_OFFLINE_PARAMS_H_
