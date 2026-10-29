# Makefile — 在 Docker 容器 (rv64-dev) 內的 /work 執行
#
#   make test          # FC + 其他運算，全部測試
#   make test-fc       # 只測 FC（KERNEL=qfc_b0 可換版本）
#   make test-ops      # 只測 GELU / add / softmax / layernorm / attention matmul
#   make imix                   # FC（small）的 instruction mix
#   make imix-op CASE=softmax   # 某個運算的 instruction mix
#   make size          # code size
#   make nofp          # 確認所有 kernel 都沒有浮點指令
#   make clean

CC      := riscv-none-elf-gcc
OBJDUMP := riscv-none-elf-objdump
NM      := riscv-none-elf-nm
SIZE    := riscv-none-elf-size
SPIKE   := spike

# Baseline 目標 ISA：RV64IMC（zicsr + zicntr 是為了讀 instret 計數器）
# 完全 bare-metal：不用 pk、不用 newlib，Spike 直接以 M-mode 執行 ELF，
# 因此 Spike 也只開 RV64IMC，任何非 IMC 指令都會被當成非法指令抓出來。
ARCH    := rv64imc_zicsr
ABI     := lp64
SPIKE_ISA := rv64imc_zicsr_zicntr

OPT     ?= -O2
CFLAGS  := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany $(OPT) -Wall \
           -ffreestanding -fno-builtin -fno-tree-loop-distribute-patterns \
           -Isrc/common -Isrc/kernels -Isrc/tests -Ibuild
LDFLAGS := -nostdlib -nostartfiles -T src/common/link.ld -lgcc -Wl,--no-warn-rwx-segments
RT      := src/common/crt0.S src/common/syscalls.c
DEPS    := $(RT) src/common/link.ld $(wildcard src/common/*.h src/kernels/*.h src/tests/*.h)

BUILD   := build

.PHONY: all test test-fc test-ops imix imix-op size nofp clean
.SECONDARY:   # 保留產生的測試資料 header，不要讓 make 自動刪掉
.SECONDEXPANSION:

$(BUILD):
	mkdir -p $@

# ======================================================================
# FC（= GEMM）
# ======================================================================
KERNEL  ?= qfc_b0
KSRC    := src/kernels/$(KERNEL).c

# 測試案例：名稱 → M K N
#   small       : 小尺寸、K 不是 8 的倍數（檢查尾端處理）
#   relu        : 含 ReLU 飽和
#   deit_fc1    : DeiT-Tiny MLP 第一層 (192→768)，取 16 個 token
#   deit_qkv    : DeiT-Tiny QKV 投影 (192→576)，取 16 個 token
CASES := small relu deit_fc1 deit_qkv
small_ARGS    := --M 4  --K 40  --N 12  --seed 1
relu_ARGS     := --M 3  --K 67  --N 9   --seed 2 --relu
deit_fc1_ARGS := --M 16 --K 192 --N 768 --seed 3
deit_qkv_ARGS := --M 16 --K 192 --N 576 --seed 4

FC_ELFS := $(foreach c,$(CASES),$(BUILD)/test_fc_$(c)_$(KERNEL).elf)

$(BUILD)/fc_%.h: scripts/gen_fc_test.py | $(BUILD)
	python3 scripts/gen_fc_test.py $($*_ARGS) -o $@

$(BUILD)/test_fc_%_$(KERNEL).elf: src/tests/test_fc.c $(KSRC) $(BUILD)/fc_%.h $(DEPS)
	$(CC) $(CFLAGS) -DTEST_DATA='"fc_$*.h"' -DKERNEL=$(KERNEL) $(RT) src/tests/test_fc.c $(KSRC) $(LDFLAGS) -o $@

test-fc: $(FC_ELFS)
	@mkdir -p results
	@rm -f results/fc_$(KERNEL).txt
	@for e in $(FC_ELFS); do \
	  echo "== $$e"; \
	  $(SPIKE) --isa=$(SPIKE_ISA) $$e | tee -a results/fc_$(KERNEL).txt | grep -E 'RESULT|mismatch'; \
	done

# ======================================================================
# 其他運算：案例名稱 → (運算種類, 產生器參數)
#   運算種類 OP 決定用哪個 kernel（src/kernels/OP_b0.c）與測試（src/tests/test_OP.c）
# ======================================================================
OP_CASES := gelu add softmax softmax_short layernorm matmul_qk matmul_pv

gelu_OP          := gelu
gelu_GEN         := --n 3072 --seed 11                 # 16 token × 192
add_OP           := add
add_GEN          := --n 3072 --seed 12                 # 16 token × 192
softmax_OP       := softmax
softmax_GEN      := --rows 8 --L 197 --seed 13         # attention 一列 = 197 個 token
softmax_short_OP := softmax
softmax_short_GEN:= --rows 16 --L 16 --seed 14        # 短列：輸出機率較大，測高位數
layernorm_OP     := layernorm
layernorm_GEN    := --rows 16 --C 192 --seed 15
matmul_qk_OP     := matmul
matmul_qk_GEN    := --M 16 --K 64 --N 197 --seed 16    # Q·Kᵀ（head dim 64）
matmul_pv_OP     := matmul
matmul_pv_GEN    := --M 16 --K 197 --N 64 --seed 17    # P·V（K=197 為奇數）

OP_ELFS := $(foreach c,$(OP_CASES),$(BUILD)/op_$(c).elf)

$(BUILD)/op_%.h: scripts/gen_ops_test.py scripts/qref.py | $(BUILD)
	python3 scripts/gen_ops_test.py $($*_OP) $($*_GEN) -o $@

$(BUILD)/op_%.elf: src/tests/test_$$($$*_OP).c src/kernels/$$($$*_OP)_b0.c $(BUILD)/op_%.h $(DEPS)
	$(CC) $(CFLAGS) -DTEST_DATA='"op_$*.h"' $(RT) src/tests/test_$($*_OP).c src/kernels/$($*_OP)_b0.c $(LDFLAGS) -o $@

test-ops: $(OP_ELFS)
	@mkdir -p results
	@rm -f results/ops_b0.txt
	@for c in $(OP_CASES); do \
	  echo "== $$c"; \
	  $(SPIKE) --isa=$(SPIKE_ISA) $(BUILD)/op_$$c.elf | tee -a results/ops_b0.txt | grep -E 'RESULT|mismatch|TRAP'; \
	done

# ======================================================================
all: $(FC_ELFS) $(OP_ELFS)
test: test-fc test-ops

# instruction mix：只統計 PC 落在 kernel 函式內的指令
imix: $(BUILD)/test_fc_small_$(KERNEL).elf
	$(SPIKE) --isa=$(SPIKE_ISA) -l $< 2>&1 >/dev/null | \
	  python3 scripts/imix.py --elf $< --func $(KERNEL)

CASE ?= softmax
imix-op: $(BUILD)/op_$(CASE).elf
	$(SPIKE) --isa=$(SPIKE_ISA) -l $< 2>&1 >/dev/null | \
	  python3 scripts/imix.py --elf $< --func q$($(CASE)_OP)_b0

size: $(BUILD)/test_fc_small_$(KERNEL).elf $(OP_ELFS)
	@echo "kernel 函式大小（bytes）："
	@for e in $^; do \
	  $(NM) -S $$e | grep -E ' q[a-z]+_b0$$' | \
	    while read addr sz t name; do echo "  $$name $$((0x$$sz))"; done; \
	done | sort -u

# 確認 kernel 裡沒有浮點指令
nofp: $(BUILD)/test_fc_small_$(KERNEL).elf $(OP_ELFS)
	@fp=0; for e in $^; do \
	  if $(OBJDUMP) -d $$e | awk '/<q[a-z]+_b0>:/,/^$$/' | grep -qE '\s(f[a-z.]+|c\.f[a-z]+)\s'; then \
	    echo "發現浮點指令：$$e"; fp=1; fi; \
	done; [ $$fp = 0 ] && echo "OK：所有 kernel 都沒有浮點指令"

clean:
	rm -rf $(BUILD)
