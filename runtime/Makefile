# runtime/Makefile — 用 LiteRT Micro 在 Spike 上執行完整 DeiT-Tiny（INT8）
# 在 Docker 容器 (rv64-dev) 的 /work/runtime 執行：
#
#   make io         # （在有 ai-edge-litert 的 Python 環境）產生輸入圖片與參考答案
#   make            # 編譯（第一次會先在電腦上產生離線參數表 build/offline_params.bin）
#   make run        # 在 Spike 上跑一次完整推論，與參考答案逐位元比對
#   make profile    # 同上，另外列出每個運算的指令數
#   make audit      # 測試：目標端照原本方式用浮點計算參數，逐筆與離線參數表比對
#   make analysis   # workload 分析（PC 直方圖 → 指令組成、熱點、code size），約 8 分鐘
#   make clean
#
# 浮點：RV64IMC 沒有浮點硬體。LiteRT Micro 原本在初始化（Prepare）時用浮點把 scale 換算成
# 整數參數；這裡改成在電腦上離線算好（tools/gen_offline_params.cc，記錄模式），
# 目標端直接讀表（重播模式），因此整個程式執行期間沒有任何浮點運算。
#   OFFLINE=2  重播（預設）    OFFLINE=3  稽核（測試用）    OFFLINE=0  原本做法（對照用）
#
# 版本：VARIANT=b0（預設，LiteRT Micro 原本的 reference kernel）
#       VARIANT=b1（軟體優化的 kernel：kernels/*_b1.cc），執行檔 build/deit_b1.elf
#   make VARIANT=b1 run / profile / analysis

CC      := riscv-none-elf-gcc
CXX     := riscv-none-elf-g++
HOSTCXX ?= g++
SPIKE   := spike
ARCH    := rv64imc_zicsr
ABI     := lp64
SPIKE_ISA := rv64imc_zicsr_zicntr

OPT     ?= -O2
OFFLINE ?= 2
VARIANT ?= b0
BUILD   := build
MODEL   := ../models/deit_tiny_int8.tflite
IO_DIR  := $(BUILD)/io
PARAMS  := $(BUILD)/offline_params.bin

# 不同 OFFLINE 模式、不同版本的目的檔分開放，避免混用
ifeq ($(VARIANT),b0)
  VAR_DEFS :=
  VAR_TAG  :=
else ifeq ($(VARIANT),b1)
  VAR_DEFS := -DDEIT_B1
  VAR_TAG  := _b1
else
  $(error VARIANT 只能是 b0 或 b1)
endif
OBJ     := $(BUILD)/obj-p$(OFFLINE)$(VAR_TAG)
ifeq ($(OFFLINE),2)
  ELF_SUFFIX := $(VAR_TAG)
else ifeq ($(OFFLINE),3)
  ELF_SUFFIX := $(VAR_TAG)_audit
else
  ELF_SUFFIX := $(VAR_TAG)_float
endif
ELF     := $(BUILD)/deit$(ELF_SUFFIX).elf

TFLM    := tflm
INCLUDES := -I. -I$(TFLM) -I$(TFLM)/third_party/flatbuffers/include -I$(TFLM)/third_party/gemmlowp \
            -I$(TFLM)/third_party/ruy -I$(TFLM)/third_party/kissfft
DEFS    := -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DNDEBUG
COMMON  := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany $(OPT) -ffunction-sections -fdata-sections
CXXFLAGS := $(COMMON) $(DEFS) $(VAR_DEFS) -DTFLM_OFFLINE_PARAMS=$(OFFLINE) -std=c++17 -fno-exceptions -fno-rtti \
            -fno-threadsafe-statics -fno-unwind-tables -fno-asynchronous-unwind-tables \
            -Wno-unused-parameter -Wno-unused-function -MMD -MP $(INCLUDES)
CFLAGS  := $(COMMON) -MMD -MP $(INCLUDES)

# 軟體浮點計數（fp_counter.c）
FP_FUNCS := __addsf3 __subsf3 __mulsf3 __divsf3 __eqsf2 __nesf2 __ltsf2 __lesf2 __gtsf2 __gesf2 __unordsf2 \
            __adddf3 __subdf3 __muldf3 __divdf3 __eqdf2 __nedf2 __ltdf2 __ledf2 __gtdf2 __gedf2 __unorddf2 \
            __fixsfsi __fixsfdi __fixunssfsi __fixunssfdi __fixdfsi __fixdfdi __fixunsdfsi __fixunsdfdi \
            __floatsisf __floatdisf __floatunsisf __floatundisf __floatsidf __floatdidf __floatunsidf __floatundidf \
            __extendsfdf2 __truncdfsf2 __negsf2 __negdf2 \
            sqrtf sqrt expf exp logf log tanhf tanh erfcf erfc roundf round \
            floorf floor ceilf ceil powf pow frexpf frexp ldexpf ldexp
null  :=
space := $(null) $(null)
comma := ,
WRAP  := -Wl,--wrap=$(subst $(space),$(comma)--wrap=,$(strip $(FP_FUNCS)))

LDFLAGS := -march=$(ARCH) -mabi=$(ABI) -nostartfiles -T bsp/link_tflm.ld -Wl,--gc-sections \
           -Wl,--no-warn-rwx-segments $(WRAP)

# LiteRT Micro 的原始碼（micro_time.cc 在目標端由 bsp/platform.cc 取代）
TFLM_ALL  := $(shell find $(TFLM)/tensorflow -name '*.cc')
TFLM_SRCS := $(filter-out %/micro_time.cc,$(TFLM_ALL))
APP_SRCS  := main.cc kernels/gelu_int8.cc bsp/platform.cc
B1_SRCS   := $(wildcard kernels/*_b1.cc)
C_SRCS    := bsp/newlib_stubs.c bsp/fp_counter.c

COMMON_OBJS := $(patsubst %.cc,$(OBJ)/%.o,$(TFLM_SRCS) kernels/gelu_int8.cc $(B1_SRCS) bsp/platform.cc) \
               $(patsubst %.c,$(OBJ)/%.o,$(C_SRCS)) $(OBJ)/crt0.o $(BUILD)/data.o
OBJS := $(COMMON_OBJS) $(OBJ)/main.o

.PHONY: all run profile dump audit analysis params io clean
all: $(ELF)

$(OBJ)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/crt0.o: ../src/common/crt0.S
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/data.o: data.S $(MODEL) $(IO_DIR)/input.bin $(IO_DIR)/golden.bin $(PARAMS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -Wa,-I$(dir $(MODEL)) -Wa,-I$(IO_DIR) -Wa,-I$(BUILD) -c $< -o $@

$(ELF): $(OBJS) bsp/link_tflm.ld
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

# ---------------------------------------------------------------- 離線參數表（電腦端）
# 用同一份 LiteRT Micro 原始碼，以記錄模式（TFLM_OFFLINE_PARAMS=1）編成電腦上的程式，
# 執行 AllocateTensors() 把每個運算 Prepare 算出的整數參數存成 build/offline_params.bin。
HOST      := $(BUILD)/host
HOSTFLAGS := -O1 -std=c++17 $(DEFS) -DTFLM_OFFLINE_PARAMS=1 -fno-exceptions -fno-rtti \
             -Wno-unused-parameter -Wno-unused-function -MMD -MP $(INCLUDES)
HOST_OBJS := $(patsubst %.cc,$(HOST)/%.o,$(TFLM_ALL) kernels/gelu_int8.cc tools/gen_offline_params.cc)

$(HOST)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(HOSTCXX) $(HOSTFLAGS) -c $< -o $@

$(HOST)/gen_offline_params: $(HOST_OBJS)
	$(HOSTCXX) $(HOST_OBJS) -o $@

$(PARAMS): $(HOST)/gen_offline_params $(MODEL)
	$(HOST)/gen_offline_params $(MODEL) $@

params: $(PARAMS)

# ---------------------------------------------------------------- 執行
run: $(ELF)
	$(SPIKE) --isa=$(SPIKE_ISA) $<

# 稽核：目標端照原本方式（軟體浮點）計算參數，逐筆與電腦算的離線參數表比對
audit:
	$(MAKE) OFFLINE=3 $(BUILD)/deit_audit.elf
	$(SPIKE) --isa=$(SPIKE_ISA) $(BUILD)/deit_audit.elf

$(BUILD)/deit$(ELF_SUFFIX)_profile.elf: $(COMMON_OBJS) main.cc bsp/link_tflm.ld
	$(CXX) $(CXXFLAGS) -DPROFILE -c main.cc -o $(OBJ)/main_profile.o
	$(CXX) $(COMMON_OBJS) $(OBJ)/main_profile.o $(LDFLAGS) -o $@

profile: $(BUILD)/deit$(ELF_SUFFIX)_profile.elf
	$(SPIKE) --isa=$(SPIKE_ISA) $< > $(BUILD)/profile_rv.txt; tail -1 $(BUILD)/profile_rv.txt
	python3 ../scripts/profile_report.py $(BUILD)/profile_rv.txt -o ../results/runtime_$(VARIANT)_profile \
	    --label "$(VARIANT)（$(if $(VAR_TAG),軟體優化 kernel,LiteRT Micro 原始 reference kernel)，-O2）"

# 除錯用：保留所有中間 tensor，印出每個運算輸出的檢查碼（make dump > build/dump_rv.txt）
$(BUILD)/deit$(ELF_SUFFIX)_dump.elf: $(COMMON_OBJS) main.cc bsp/link_tflm.ld
	$(CXX) $(CXXFLAGS) -DDUMP -DARENA_SIZE=268435456 -c main.cc -o $(OBJ)/main_dump.o
	$(CXX) $(COMMON_OBJS) $(OBJ)/main_dump.o $(LDFLAGS) -o $@

dump: $(BUILD)/deit$(ELF_SUFFIX)_dump.elf
	$(SPIKE) --isa=$(SPIKE_ISA) $<

# ---------------------------------------------------------------- workload 分析（規格 2.B）
# deit_noinvoke.elf：同一份程式，只把 g_do_invoke 設成 0（程式碼位址完全相同），
# 兩份 PC 直方圖相減 = 只屬於推論的指令。（連結順序必須與 deit.elf 相同）
$(BUILD)/deit$(ELF_SUFFIX)_noinvoke.elf: $(COMMON_OBJS) main.cc bsp/link_tflm.ld
	$(CXX) $(CXXFLAGS) -DDO_INVOKE=0 -c main.cc -o $(OBJ)/main_noinvoke.o
	$(CXX) $(COMMON_OBJS) $(OBJ)/main_noinvoke.o $(LDFLAGS) -o $@

analysis: $(ELF) $(BUILD)/deit$(ELF_SUFFIX)_noinvoke.elf
	$(SPIKE) -g --isa=$(SPIKE_ISA) $(ELF) > $(BUILD)/hist.txt 2>&1
	$(SPIKE) -g --isa=$(SPIKE_ISA) $(BUILD)/deit$(ELF_SUFFIX)_noinvoke.elf > $(BUILD)/hist_noinvoke.txt 2>&1
	riscv-none-elf-objdump -d -C $(ELF) > $(BUILD)/deit.dis
	riscv-none-elf-size -A $(ELF) > $(BUILD)/size.txt
	python3 ../scripts/hot_loops.py $(BUILD)/hist.txt $(BUILD)/deit.dis -o ../results/hot_loops_b0.md
	python3 ../scripts/check_nofp.py $(BUILD)/hist.txt $(BUILD)/deit.dis
	python3 ../scripts/workload_analysis.py --hist $(BUILD)/hist.txt --hist-base $(BUILD)/hist_noinvoke.txt \
	    --dis $(BUILD)/deit.dis --size $(BUILD)/size.txt --macs ../results/model_macs.txt \
	    --profile ../results/runtime_b0_profile_ops.csv -o ../results/workload_b0

# 產生輸入圖片（INT8）與 Python LiteRT 參考答案；需要 ai-edge-litert、pillow
io:
	python3 ../scripts/make_runtime_io.py $(MODEL) $(IO_DIR)

clean:
	rm -rf $(BUILD)/obj-p* $(BUILD)/host $(BUILD)/*.elf $(BUILD)/data.o $(BUILD)/data.d $(PARAMS)

-include $(OBJS:.o=.d) $(HOST_OBJS:.o=.d)
