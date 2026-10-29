# RV64IMC Quantized NN — Baseline Kernels

EE6455 高等計算機結構 term project（CV-Wally / Spike）。

## 環境

```bash
docker build -t rv64-dev .                          # 只需一次
docker run --rm -it -v "$PWD":/work rv64-dev bash   # 進入容器，以下指令都在容器內執行
```

## 執行

```bash
make test                   # 全部測試（FC + 其他運算），逐位元比對 golden
make test-fc                # 只測 FC
make test-ops               # 只測 GELU / add / softmax / layernorm / attention matmul
make imix                   # FC（small）的 instruction mix
make imix-op CASE=softmax   # 其他運算的 instruction mix（CASE 見 Makefile 的 OP_CASES）
make size                   # 各 kernel 的 code size
make nofp                   # 確認所有 kernel 都沒有浮點指令
make clean
```

換測試版本：`make test KERNEL=qfc_b1`（之後新增 B1 / X1 kernel 時使用）。

## 結構

```text
src/common/   bare-metal 執行環境：crt0.S、link.ld、syscalls.c（HTIF printf/exit）、perf.h（rdinstret）
src/kernels/  requant.h（TFLite 相容 requant，共用）
              qfc.h / qfc_b0.c（FC）
              ops.h / gelu_b0.c / add_b0.c / softmax_b0.c / layernorm_b0.c / matmul_b0.c
src/tests/    test_fc.c、test_<op>.c、test_common.h：執行 kernel、量 instret、逐位元比對
scripts/      qref.py（與 C 逐位元一致的整數參考實作）
              gen_fc_test.py、gen_ops_test.py（測試資料 + golden + 整數近似誤差）
              imix.py（trace → instruction mix）
results/      make test 的輸出
```

## 設計重點

- **完全 bare-metal**：不用 pk / newlib，Spike 以 M-mode 直接執行 ELF。
  Spike ISA 設成 `rv64imc_zicsr_zicntr`，任何超出 RV64IMC 的指令都會觸發非法指令例外。
- **量化語意與 TFLite/LiteRT 相同**：int8 輸入（有 zero point）、int8 對稱權重、int32 bias、
  per-channel `(quant_mult, shift)` requant、飽和到 `[act_min, act_max]`。
- **驗證**：golden 由純 Python 獨立計算，與 C kernel 逐位元比對；不一致會印出位置並以非 0 結束。
- **量測**：kernel 前後讀 `instret`，只計 kernel 本身；instruction mix 以 kernel 函式的位址範圍過濾 trace。

## B0 結果（-O2）

| 測試 | M×K×N | 動態指令 | MAC/inst | inst/output |
|---|---|---:|---:|---:|
| small | 4×40×12 | 18,112 | 0.106 | 377 |
| relu | 3×67×9 | 16,061 | 0.112 | 594 |
| deit_fc1 | 16×192×768 | 19,556,686 | 0.120 | 1,591 |
| deit_qkv | 16×192×576 | 14,667,566 | 0.120 | 1,591 |

內迴圈每個 MAC 要 8 條指令：`lb ×2`、`addi ×2`（指標遞增）、`subw`（減 zero point）、`mulw`、`addw`、`bne`。

## 其他運算（B0）

| 運算 | 整數做法 | 整數近似 vs 浮點 | 測試尺寸 | inst / 輸出 |
|---|---|---|---|---:|
| GELU | 256 項查表（離線用浮點建表） | 0 LSB（由構造保證） | 3072 | 7.0 |
| 殘差相加 | TFLite ADD：兩輸入各自 requant → 相加 → requant | 0 LSB | 3072 | 111.5 |
| Softmax | 減最大值 → Q16 exp 查表 → Σ → 整數除法；輸出 scale 1/256、zp −128 | 0 LSB | 8×197、16×16 | ~30 |
| LayerNorm | 整數平均/變異數（D = C·(x−mean) 精確）→ 逐位元 isqrt → γ/β 以 Q16 定點 | ≤ 1 LSB | 16×192 | 46.8 |
| Attention matmul | 兩邊都有 zero point 的 GEMM，per-tensor requant | — | 16×64×197（Q·Kᵀ）、16×197×64（P·V） | 9.7 / MAC |

「整數近似 vs 浮點」＝整數演算法的結果與「浮點算完再量化」相比的差距，衡量近似本身的誤差；
C kernel 則必須與 Python 整數版（golden）**逐位元相同**。

### 推估：一個 DeiT-Tiny block 的 B0 指令分布

以上表的每單位指令數，乘上 DeiT-Tiny 一個 block（197 token）的實際運算量：

| 運算 | 動態指令（推估） | 比例 |
|---|---:|---:|
| FC（QKV、Proj、FC1、FC2） | 722 M | 81.7% |
| Attention matmul（Q·Kᵀ、P·V） | 145 M | 16.4% |
| 殘差相加 | 8.4 M | 1.0% |
| Softmax | 3.5 M | 0.4% |
| LayerNorm | 3.5 M | 0.4% |
| GELU | 1.1 M | 0.1% |
| **合計** | **~884 M / block，×12 ≈ 10.6 G** | |

GEMM 類佔 98%，是 Xqmac8 的主要目標。殘差相加每個元素要做 3 次 requant，
在 GEMM 被加速後會變成明顯的比例，也是 requant 指令的受益對象。

**尚未實作**：patch embedding（16×16、stride 16 的 conv，patch 不重疊，可攤平後直接用 `qfc_b0`）、
把各 kernel 串成完整 block、真實 DeiT 權重。
