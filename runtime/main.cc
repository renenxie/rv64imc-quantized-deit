// main.cc — 用 LiteRT Micro 在 bare-metal RV64IMC（Spike）上執行完整 DeiT-Tiny 推論
//
// 流程：載入模型 → 配置記憶體 → 放入 INT8 圖片 → Invoke（整個 DeiT）→ 與 Python 參考答案比對
// 量測：Invoke 前後讀 instret，就是一次完整推論的動態指令數
//       （可選）MicroProfiler 以 instret 為 tick，列出每個運算各花多少指令
// 浮點：預設以離線參數表重播（TFLM_OFFLINE_PARAMS=2），初始化與推論都不做任何浮點運算；
//       fp_counter.c 統計整個程式的軟體浮點呼叫次數，必須是 0 才算 PASS。
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/offline_params.h"
#include "op_resolver.h"

// 由 data.S 以 .incbin 嵌入
extern "C" const uint8_t g_model[];
extern "C" const uint8_t g_input[], g_input_end[];
extern "C" const uint8_t g_golden[], g_golden_end[];
extern "C" void __libc_init_array(void);
extern "C" uint64_t g_fp_calls;  // fp_counter.c：軟體浮點函式被呼叫的次數

static inline uint64_t rdinstret() {
  uint64_t x;
  __asm__ volatile("rdinstret %0" : "=r"(x));
  return x;
}

#ifdef PROFILE
// 以 instret 為計數的 profiler：記錄每個運算（依執行順序）花了多少條指令。
// LiteRT Micro 內建的 MicroProfiler 用 32 位元累加，整個 DeiT 會溢位，所以自己寫一個 64 位元版本。
class InstretProfiler : public tflite::MicroProfilerInterface {
 public:
  static constexpr int kMaxEvents = 1024;
  uint32_t BeginEvent(const char* tag) override {
    if (n_ >= kMaxEvents) return kMaxEvents;
    tag_[n_] = tag;
    start_[n_] = rdinstret();
    return n_++;
  }
  void EndEvent(uint32_t h) override {
    if (h < kMaxEvents) end_[h] = rdinstret();
  }
  // 輸出格式（scripts/profile_report.py 會讀）：
  //   OP,<運算編號>,<種類>,<指令數>
  //   TAG,<種類>,<次數>,<指令數>
  //   TOTAL,<所有運算指令數>
  void Print() const {
    printf("---- PROFILE（每個運算的指令數，instret）----\n");
    uint64_t total = 0;
    for (int i = 0; i < n_; ++i) {
      uint64_t d = end_[i] - start_[i];
      total += d;
      printf("OP,%d,%s,%lu\n", i, tag_[i], (unsigned long)d);
    }
    for (int i = 0; i < n_; ++i) {
      bool first = true;
      for (int j = 0; j < i; ++j)
        if (strcmp(tag_[i], tag_[j]) == 0) { first = false; break; }
      if (!first) continue;
      uint64_t sum = 0;
      int cnt = 0;
      for (int j = i; j < n_; ++j)
        if (strcmp(tag_[i], tag_[j]) == 0) { sum += end_[j] - start_[j]; ++cnt; }
      printf("TAG,%s,%d,%lu\n", tag_[i], cnt, (unsigned long)sum);
    }
    printf("TOTAL,%lu\n", (unsigned long)total);
    printf("---- END PROFILE ----\n");
  }

 private:
  const char* tag_[kMaxEvents];
  uint64_t start_[kMaxEvents], end_[kMaxEvents];
  int n_ = 0;
};
#endif

// 分析用：g_do_invoke = 0 時跳過推論，只做初始化與前後處理（make analysis 用來扣掉非推論的指令）。
// 只改這個變數的初始值，程式碼位址完全不變，兩個版本的 PC 直方圖可以直接相減。
#ifndef DO_INVOKE
#define DO_INVOKE 1
#endif
// 固定放在 .data（值為 0 時編譯器會把它移到 .bss，位址就會變）
__attribute__((section(".data"))) volatile int g_do_invoke = DO_INVOKE;

#ifndef ARENA_SIZE
#define ARENA_SIZE (8 * 1024 * 1024)
#endif
alignas(16) static uint8_t g_arena[ARENA_SIZE];

int main() {
  __libc_init_array();  // 執行 C++ 靜態建構子

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("模型 schema 版本不符：%lu\n", (unsigned long)model->version());
    return 2;
  }

  // 只註冊 DeiT 用到的運算（op_resolver.h，與電腦端產生離線參數的工具共用）
  static DeitOpResolver resolver;
  RegisterDeitOps(resolver);

#if defined(PROFILE)
  static InstretProfiler profiler;
  tflite::MicroInterpreter interpreter(model, resolver, g_arena, ARENA_SIZE, nullptr, &profiler);
#elif defined(DUMP)
  // 除錯用：保留所有中間 tensor，推論後逐一列出每個運算輸出的檢查碼
  tflite::MicroInterpreter interpreter(model, resolver, g_arena, ARENA_SIZE, nullptr, nullptr, true);
#else
  tflite::MicroInterpreter interpreter(model, resolver, g_arena, ARENA_SIZE);
#endif

  uint64_t a0 = rdinstret(), f0 = g_fp_calls;
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors 失敗\n");
    return 3;
  }
  uint64_t a1 = rdinstret(), f1 = g_fp_calls;
  printf("初始化（AllocateTensors）：%lu 條指令，軟體浮點呼叫 %lu 次\n",
         (unsigned long)(a1 - a0), (unsigned long)(f1 - f0));
  printf("arena 使用量：%lu bytes（配置 %d）\n",
         (unsigned long)interpreter.arena_used_bytes(), ARENA_SIZE);
#if TFLM_OFFLINE_PARAMS == 2
  printf("初始化參數：讀離線參數表（offline_params.bin），不做浮點計算\n");
#elif TFLM_OFFLINE_PARAMS == 3
  printf("離線參數稽核：比對 %u 筆，不一致 %u 筆（%s）\n",
         (unsigned)tflite::micro::offline::AuditChecked(),
         (unsigned)tflite::micro::offline::AuditMismatches(),
         tflite::micro::offline::AuditMismatches() == 0 ? "PASS" : "FAIL");
#else
  printf("初始化參數：Prepare 時用（軟體）浮點計算\n");
#endif

  TfLiteTensor* in = interpreter.input(0);
  TfLiteTensor* out = interpreter.output(0);
  const size_t in_bytes = (size_t)(g_input_end - g_input);
  const size_t gold_bytes = (size_t)(g_golden_end - g_golden);
  if (in->type != kTfLiteInt8 || in->bytes != in_bytes || out->bytes != gold_bytes) {
    printf("輸入/輸出大小不符：in %lu vs %lu，out %lu vs %lu\n", (unsigned long)in->bytes,
           (unsigned long)in_bytes, (unsigned long)out->bytes, (unsigned long)gold_bytes);
    return 4;
  }
  memcpy(in->data.int8, g_input, in_bytes);

  uint64_t i0 = 0, i1 = 0, p0 = 0, p1 = 0;
  TfLiteStatus st = kTfLiteOk;
  if (g_do_invoke) {
    i0 = rdinstret(), p0 = g_fp_calls;
    st = interpreter.Invoke();
    i1 = rdinstret(), p1 = g_fp_calls;
  }
  if (st != kTfLiteOk) {
    printf("Invoke 失敗\n");
    return 5;
  }

#ifdef PROFILE
  profiler.Print();
#endif

#ifdef DUMP
  // 每個運算的輸出：op 編號、tensor 編號、bytes、FNV-1a 檢查碼（與 scripts/dump_ref_tensors.py 對照）
  {
    const auto* ops = model->subgraphs()->Get(0)->operators();
    for (unsigned i = 0; i < ops->size(); ++i) {
      int t = ops->Get(i)->outputs()->Get(0);
      TfLiteEvalTensor* et = interpreter.GetTensor(t);
      size_t n = 0;
      tflite::TfLiteEvalTensorByteLength(et, &n);
      const uint8_t* d = reinterpret_cast<const uint8_t*>(et->data.raw);
      uint32_t h = 2166136261u;
      for (size_t k = 0; k < n; ++k) h = (h ^ d[k]) * 16777619u;
      printf("T %u %d %lu %08x\n", i, t, (unsigned long)n, (unsigned)h);
    }
  }
#endif

  // 與 Python LiteRT 參考答案逐位元比對
  const int8_t* y = out->data.int8;
  const int8_t* g = reinterpret_cast<const int8_t*>(g_golden);
  int mismatch = 0, maxdiff = 0;
  for (size_t i = 0; i < gold_bytes; ++i) {
    int d = y[i] - g[i];
    if (d < 0) d = -d;
    if (d) ++mismatch;
    if (d > maxdiff) maxdiff = d;
  }
  int top = 0, gtop = 0;
  for (size_t i = 1; i < gold_bytes; ++i) {
    if (y[i] > y[top]) top = (int)i;
    if (g[i] > g[gtop]) gtop = (int)i;
  }

  // 浮點檢查：推論中必須是 0；重播模式（預設）下整個程式（從開機、C++ 靜態建構子、初始化、
  // 推論到比對）也必須是 0。g_fp_calls 是 .bss 變數，開機時為 0。
  const uint64_t fp_total = g_fp_calls;
#if TFLM_OFFLINE_PARAMS == 2
  const bool fp_ok = fp_total == 0;
#else
  const bool fp_ok = p1 == p0;
#endif
  const bool pass = mismatch == 0 && fp_ok;
  printf("RESULT model=deit_tiny_int8 invoke_insts=%lu fp_calls_in_invoke=%lu fp_calls_total=%lu "
         "top1=%d golden_top1=%d mismatches=%d maxdiff=%d status=%s\n",
         (unsigned long)(i1 - i0), (unsigned long)(p1 - p0), (unsigned long)fp_total, top, gtop,
         mismatch, maxdiff, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
