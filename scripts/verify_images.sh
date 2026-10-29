#!/usr/bin/env bash
# verify_images.sh — 用多張圖片驗證 RISC-V 上的完整模型與 Python 參考答案逐位元相同
#
#   1.（Mac 的 Python 環境，需要 ai-edge-litert）每張圖產生輸入與參考答案：
#        for i in 0 1 2 ...; do
#          python3 scripts/make_runtime_io.py models/deit_tiny_int8.tflite runtime/build/io_multi/$i --index $i
#        done
#   2.（容器內）bash /work/scripts/verify_images.sh [SPIKE 路徑]
#
# 每張圖：把 runtime/build/io_multi/<編號>/ 嵌進執行檔、在 Spike 上跑一次，收集 RESULT。
# 結束後恢復成預設的 build/io（make run 用的那張圖）。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="$ROOT/runtime"
SPIKE="${1:-spike}"
cd "$RT"
pass=0; fail=0
printf "%-6s %-14s %-6s %-6s %-10s %s\n" 圖片 invoke_insts top1 golden mismatch status
for d in $(ls -d build/io_multi/*/ 2>/dev/null | sort -t/ -k3 -n); do
  idx=$(basename "$d")
  rm -f build/data.o
  make -s IO_DIR="${d%/}" build/deit.elf >/dev/null || { echo "$idx 編譯失敗"; fail=$((fail+1)); continue; }
  r=$("$SPIKE" --isa=rv64imc_zicsr_zicntr build/deit.elf 2>&1 | grep '^RESULT')
  get() { echo "$r" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
  st=$(get status)
  printf "%-6s %-14s %-6s %-6s %-10s %s\n" "$idx" "$(get invoke_insts)" "$(get top1)" "$(get golden_top1)" "$(get mismatches)" "$st"
  [ "$st" = PASS ] && pass=$((pass+1)) || fail=$((fail+1))
done
rm -f build/data.o && make -s build/deit.elf >/dev/null   # 恢復預設圖片
echo "共 $((pass+fail)) 張：PASS $pass、FAIL $fail"
[ "$fail" -eq 0 ]
