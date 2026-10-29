// [期中project 新增] offline_params.cc — 離線參數表的記錄 / 重播 / 稽核（說明見 offline_params.h）
//
// 離線表格式（little-endian）：
//   "TFOP"、uint32 筆數，接著每筆：uint16 op、uint16 slot、uint32 bytes、資料（補齊到 4 的倍數）
#include "tensorflow/lite/micro/offline_params.h"

#include <cstring>

#include "tensorflow/lite/micro/micro_log.h"

#if TFLM_OFFLINE_PARAMS == 2 || TFLM_OFFLINE_PARAMS == 3
// 目標端：離線表由 data.S 以 .incbin 嵌入
extern "C" const uint8_t g_offline_params[];
extern "C" const uint8_t g_offline_params_end[];
#endif

namespace tflite {
namespace micro {
namespace offline {
namespace {

int g_current_op = -1;
uint32_t g_audit_checked = 0;
uint32_t g_audit_mismatch = 0;

#if TFLM_OFFLINE_PARAMS == 1
constexpr size_t kMaxTable = 4 * 1024 * 1024;
uint8_t g_table[kMaxTable];
size_t g_table_bytes = 8;  // 先保留表頭
uint32_t g_records = 0;
#endif

uint32_t Rd32(const uint8_t* p) {
  return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t Rd16(const uint8_t* p) { return p[0] | (p[1] << 8); }

#if TFLM_OFFLINE_PARAMS == 2 || TFLM_OFFLINE_PARAMS == 3
// 在離線表中找 (op, slot)；回傳資料位置與大小
const uint8_t* Find(int op, int slot, uint32_t* bytes) {
  const uint8_t* p = g_offline_params;
  const uint8_t* end = g_offline_params_end;
  if (end - p < 8 || std::memcmp(p, "TFOP", 4) != 0) return nullptr;
  const uint32_t n = Rd32(p + 4);
  p += 8;
  for (uint32_t i = 0; i < n && p + 8 <= end; ++i) {
    const int rop = Rd16(p);
    const int rslot = Rd16(p + 2);
    const uint32_t rbytes = Rd32(p + 4);
    if (rop == op && rslot == slot) {
      *bytes = rbytes;
      return p + 8;
    }
    p += 8 + ((rbytes + 3) & ~3u);
  }
  return nullptr;
}
#endif

}  // namespace

void SetCurrentOp(int op_index) { g_current_op = op_index; }
int CurrentOp() { return g_current_op; }

void Save(int slot, const void* data, size_t bytes) {
#if TFLM_OFFLINE_PARAMS == 1
  const size_t need = 8 + ((bytes + 3) & ~static_cast<size_t>(3));
  if (g_table_bytes + need > kMaxTable) {
    MicroPrintf("offline params: 表格太大");
    return;
  }
  uint8_t* p = g_table + g_table_bytes;
  p[0] = g_current_op & 0xff;
  p[1] = (g_current_op >> 8) & 0xff;
  p[2] = slot & 0xff;
  p[3] = (slot >> 8) & 0xff;
  p[4] = bytes & 0xff;
  p[5] = (bytes >> 8) & 0xff;
  p[6] = (bytes >> 16) & 0xff;
  p[7] = (bytes >> 24) & 0xff;
  std::memset(p + 8, 0, need - 8);
  std::memcpy(p + 8, data, bytes);
  g_table_bytes += need;
  ++g_records;
#elif TFLM_OFFLINE_PARAMS == 3
  uint32_t n = 0;
  const uint8_t* rec = Find(g_current_op, slot, &n);
  ++g_audit_checked;
  if (rec == nullptr || n != bytes || std::memcmp(rec, data, bytes) != 0) {
    ++g_audit_mismatch;
    MicroPrintf("offline params 稽核不一致：op %d slot %d", g_current_op, slot);
  }
#else
  (void)slot;
  (void)data;
  (void)bytes;
#endif
}

TfLiteStatus Load(TfLiteContext* context, int slot, void* data, size_t bytes) {
#if TFLM_OFFLINE_PARAMS == 2
  uint32_t n = 0;
  const uint8_t* rec = Find(g_current_op, slot, &n);
  if (rec == nullptr || n != bytes) {
    MicroPrintf("offline params: 找不到 op %d slot %d（或大小不符）", g_current_op, slot);
    return kTfLiteError;
  }
  std::memcpy(data, rec, bytes);
  return kTfLiteOk;
#else
  (void)context;
  (void)slot;
  (void)data;
  (void)bytes;
  MicroPrintf("offline params: 只有重播模式可以 Load");
  return kTfLiteError;
#endif
}

const uint8_t* RecordedTable(size_t* bytes) {
#if TFLM_OFFLINE_PARAMS == 1
  std::memcpy(g_table, "TFOP", 4);
  g_table[4] = g_records & 0xff;
  g_table[5] = (g_records >> 8) & 0xff;
  g_table[6] = (g_records >> 16) & 0xff;
  g_table[7] = (g_records >> 24) & 0xff;
  *bytes = g_table_bytes;
  return g_table;
#else
  *bytes = 0;
  return nullptr;
#endif
}

uint32_t AuditChecked() { return g_audit_checked; }
uint32_t AuditMismatches() { return g_audit_mismatch; }

}  // namespace offline
}  // namespace micro
}  // namespace tflite
