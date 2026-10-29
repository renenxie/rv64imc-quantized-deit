#!/usr/bin/env bash
# build_spike.sh — 編譯專案裡的 Spike（spike/，之後要在這份原始碼加新指令）
#
#   在容器內：bash /work/scripts/build_spike.sh          # 第一次約 5～10 分鐘
#   改了 Spike 原始碼後再執行一次：只重編改到的部分
#   產生：spike/build/spike
#   使用：/work/spike/build/spike --isa=rv64imc_zicsr_zicntr <程式.elf>
#         或 make run SPIKE=/work/spike/build/spike
#
# 編譯目錄 spike/build/ 不進 git（.gitignore 的 build/）。
# 需要：g++（C++20）、make、dtc —— rv64-dev 容器裡都有。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/spike"
BLD="$SRC/build"
JOBS="${JOBS:-$( (nproc 2>/dev/null || echo 4) )}"

[ -f "$SRC/configure" ] || { echo "找不到 $SRC/configure"; exit 1; }
mkdir -p "$BLD"
cd "$BLD"
if [ ! -f Makefile ]; then
  echo "==> configure（第一次）"
  "$SRC/configure" --prefix="$BLD/install" \
      --without-boost --without-boost-asio --without-boost-regex > configure.log
fi
echo "==> make -j$JOBS"
make -j"$JOBS" spike > make.log 2>&1 || { tail -30 make.log; echo "編譯失敗，完整紀錄：$BLD/make.log"; exit 1; }
echo "完成：$BLD/spike"
"$BLD/spike" --help 2>&1 | head -1
