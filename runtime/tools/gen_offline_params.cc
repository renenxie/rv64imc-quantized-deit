// gen_offline_params.cc — 在電腦（x86）上產生離線參數表（offline_params.bin）
//
//   （由 runtime/Makefile 的 `make params` 編譯、執行）
//   build/host/gen_offline_params ../models/deit_tiny_int8.tflite build/offline_params.bin
//
// 做法：用「同一份」LiteRT Micro 原始碼（TFLM_OFFLINE_PARAMS=1，記錄模式）在電腦上執行
// AllocateTensors()，每個運算的 Prepare 照原本方式用浮點算出整數參數，並記錄下來。
// RISC-V 端以重播模式（TFLM_OFFLINE_PARAMS=2）直接讀這份表，因此目標端完全不需要浮點。
// 浮點計算只在這裡（離線、電腦上）發生，與量化模型的 Python 轉檔屬於同一類的離線前處理。
#include <cstdio>
#include <cstring>
#include <vector>

#include "op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/offline_params.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "用法：%s <model.tflite> <offline_params.bin>\n", argv[0]);
    return 2;
  }
  std::FILE* f = std::fopen(argv[1], "rb");
  if (!f) {
    std::perror(argv[1]);
    return 2;
  }
  std::vector<uint8_t> model_data;
  uint8_t buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) model_data.insert(model_data.end(), buf, buf + n);
  std::fclose(f);
  // flatbuffer 需要對齊
  std::vector<uint64_t> aligned((model_data.size() + 7) / 8);
  std::memcpy(aligned.data(), model_data.data(), model_data.size());

  const tflite::Model* model = tflite::GetModel(aligned.data());
  static DeitOpResolver resolver;
  RegisterDeitOps(resolver);
  static std::vector<uint8_t> arena(64 * 1024 * 1024);
  tflite::MicroInterpreter interpreter(model, resolver, arena.data(), arena.size());
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    std::fprintf(stderr, "AllocateTensors 失敗\n");
    return 3;
  }
  size_t bytes = 0;
  const uint8_t* table = tflite::micro::offline::RecordedTable(&bytes);
  const uint32_t records = table[4] | (table[5] << 8) | (table[6] << 16) | (table[7] << 24);
  std::FILE* o = std::fopen(argv[2], "wb");
  if (!o || std::fwrite(table, 1, bytes, o) != bytes) {
    std::perror(argv[2]);
    return 4;
  }
  std::fclose(o);
  std::printf("離線參數表：%u 筆，%zu bytes → %s\n", records, bytes, argv[2]);
  return 0;
}
