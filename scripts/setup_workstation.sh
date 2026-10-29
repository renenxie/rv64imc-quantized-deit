#!/usr/bin/env bash
# setup_workstation.sh — 在沒有 root 權限的 Linux 工作站上安裝本專案需要的工具
#
#   bash scripts/setup_workstation.sh            # 預設安裝到 ~/riscv-tools
#   RV_PREFIX=/path/to/dir bash scripts/setup_workstation.sh
#
# 安裝內容（與 Dockerfile 相同的工具，全部裝在使用者目錄，不需要 sudo）：
#   1. RISC-V GCC（xPack 預編譯版）
#   2. dtc（device tree compiler，Spike 執行時需要；工作站已有就略過）
#   3. Spike（從原始碼編譯，固定在同一個 commit；原始碼保留，之後要加 Xqmac8）
# 不安裝 pk：本專案是 bare-metal，Spike 直接執行 ELF。
#
# 重複執行是安全的：已經裝好的部分會略過。
set -euo pipefail

PREFIX="${RV_PREFIX:-$HOME/riscv-tools}"
XPACK_VER="15.2.0-1"
SPIKE_REPO="https://github.com/riscv-software-src/riscv-isa-sim.git"
SPIKE_COMMIT="7ab2efd6785e847c6d13d810c0b25b7c9c26bd21"   # 2026-09-25，已驗證可跑本專案全部測試
DTC_REPO="https://git.kernel.org/pub/scm/utils/dtc/dtc.git"
JOBS="${JOBS:-$( (nproc 2>/dev/null || echo 4) | awk '{print ($1>16)?16:$1}')}"  # 共用工作站，最多用 16 核

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mOK\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m錯誤：%s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. 檢查基本工具
say "檢查基本工具"
for t in git make tar python3; do
  command -v "$t" >/dev/null || die "找不到 $t，請先 module load 或請管理員安裝"
  ok "$t"
done
if command -v curl >/dev/null; then DL="curl -fsSL"; elif command -v wget >/dev/null; then DL="wget -qO-"
else die "需要 curl 或 wget"; fi
ok "下載工具：${DL%% *}"

command -v g++ >/dev/null || die "找不到 g++（編譯 Spike 需要）。試試 'module avail gcc' 再 'module load gcc/<版本>'"
GXX_MAJOR=$(g++ -dumpversion | cut -d. -f1)
if [ "$GXX_MAJOR" -lt 11 ]; then
  die "g++ 版本是 $(g++ -dumpversion)，Spike 需要支援 C++20 的 g++ 11 以上。
     試試 'module avail gcc'，載入較新版本後再執行一次本腳本"
fi
ok "g++ $(g++ -dumpversion)"

case "$(uname -m)" in
  x86_64)  XARCH=x64 ;;
  aarch64) XARCH=arm64 ;;
  *) die "不支援的 CPU 架構：$(uname -m)" ;;
esac
ok "架構 $(uname -m)"
ok "安裝位置 $PREFIX（平行編譯 $JOBS 核）"
mkdir -p "$PREFIX/src"

# ---------------------------------------------------------------- 1. RISC-V GCC
say "1/3 RISC-V GCC（xPack $XPACK_VER）"
GCC_DIR="$PREFIX/xpack-riscv-none-elf-gcc-$XPACK_VER"
if [ -x "$GCC_DIR/bin/riscv-none-elf-gcc" ]; then
  ok "已安裝，略過"
else
  $DL "https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v${XPACK_VER}/xpack-riscv-none-elf-gcc-${XPACK_VER}-linux-${XARCH}.tar.gz" \
    | tar xz -C "$PREFIX"
  ok "完成"
fi

# ---------------------------------------------------------------- 2. dtc
say "2/3 dtc"
if command -v dtc >/dev/null; then
  ok "工作站已有 $(command -v dtc)，略過"
elif [ -x "$PREFIX/bin/dtc" ]; then
  ok "已安裝，略過"
else
  command -v flex >/dev/null && command -v bison >/dev/null \
    || die "工作站沒有 dtc，自行編譯又需要 flex 和 bison。請試 'module avail' 找 dtc/flex/bison，或請管理員安裝 device-tree-compiler"
  rm -rf "$PREFIX/src/dtc"
  git clone -q --depth 1 "$DTC_REPO" "$PREFIX/src/dtc"
  make -C "$PREFIX/src/dtc" -j"$JOBS" NO_PYTHON=1 NO_YAML=1 NO_VALGRIND=1 PREFIX="$PREFIX" install-bin >/dev/null
  ok "完成"
fi

# ---------------------------------------------------------------- 3. Spike
say "3/3 Spike（commit ${SPIKE_COMMIT:0:12}）"
SPIKE_SRC="$PREFIX/src/riscv-isa-sim"
if [ -x "$PREFIX/bin/spike" ] && [ "$(git -C "$SPIKE_SRC" rev-parse HEAD 2>/dev/null)" = "$SPIKE_COMMIT" ]; then
  ok "已安裝，略過"
else
  if [ ! -d "$SPIKE_SRC/.git" ]; then
    git init -q "$SPIKE_SRC"
    git -C "$SPIKE_SRC" remote add origin "$SPIKE_REPO"
  fi
  git -C "$SPIKE_SRC" fetch -q --depth 1 origin "$SPIKE_COMMIT"
  git -C "$SPIKE_SRC" checkout -q FETCH_HEAD
  mkdir -p "$SPIKE_SRC/build"
  LOG="$PREFIX/src/spike-build.log"
  echo "    編譯中（約 5～15 分鐘），紀錄寫在 $LOG ..."
  ( cd "$SPIKE_SRC/build" \
    && ../configure --prefix="$PREFIX" --without-boost --without-boost-asio --without-boost-regex \
    && make -j"$JOBS" && make install ) > "$LOG" 2>&1 \
    || die "Spike 編譯失敗，請看 $LOG 最後幾十行"
  ok "完成，原始碼保留在 $SPIKE_SRC"
fi

# ---------------------------------------------------------------- 環境變數
ENV_FILE="$PREFIX/env.sh"
cat > "$ENV_FILE" <<EOF
# 由 setup_workstation.sh 產生（bash/zsh）：source 這個檔案即可使用 RISC-V 工具
export PATH="$GCC_DIR/bin:$PREFIX/bin:\$PATH"
EOF
cat > "$PREFIX/env.csh" <<EOF
# 由 setup_workstation.sh 產生（tcsh/csh）：source 這個檔案即可使用 RISC-V 工具
setenv PATH "$GCC_DIR/bin:$PREFIX/bin:\$PATH"
EOF

# ---------------------------------------------------------------- 驗證
say "驗證"
# shellcheck disable=SC1090
source "$ENV_FILE"
riscv-none-elf-gcc --version | head -1 | sed 's/^/    /'
command -v spike | sed 's/^/    spike: /'
command -v dtc   | sed 's/^/    dtc:   /'

cat <<EOF

全部完成。接下來：

  1. 讓每次登入都能用（只需做一次；先用 'echo \$SHELL' 看登入 shell）：
       bash：echo 'source $ENV_FILE' >> ~/.bashrc
       tcsh：echo 'source $PREFIX/env.csh' >> ~/.cshrc
     然後重新登入，或先手動 source 一次。

  2. 到專案資料夾跑測試，結果應該跟 Docker 裡完全一樣：
       make test
EOF
