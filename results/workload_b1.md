# DeiT-Tiny INT8 workload 分析 — b1（軟體優化 kernel，RV64IMC，-O2）

資料來源：Spike PC 直方圖（`spike -g`），扣掉只做初始化的版本後，只留下推論（Invoke）的指令。

## 1. 總覽

| 項目 | 數值 |
|---|---:|
| 推論動態指令數（直方圖相減） | 4,360,277,319 |
| 推論動態指令數（程式內 instret） | 4,360,277,226 |
| 模型 MAC 數 | 1,253,683,200 |
| **MAC / 指令** | **0.2875**（每個 MAC 平均 3.48 條指令） |
| 記憶體指令（load + store） | 1,052,694,230（24.14%） |
| load / MAC | 0.817 |
| store / MAC | 0.023 |

兩種量法差 93 條（0.00%），來自 Invoke 前後讀計數器與比對程式的些微差異，可忽略。

## 2. 指令組成（instruction mix）

| 類別 | 指令數 | 佔比 |
|---|---:|---:|
| load | 1,024,124,971 | 23.49% |
| store | 28,569,259 | 0.66% |
| 整數 ALU | 1,916,058,578 | 43.94% |
| 乘法 | 1,272,569,645 | 29.19% |
| 除法/餘數 | 299 | 0.00% |
| 條件分支 | 118,561,948 | 2.72% |
| 跳躍/呼叫 | 392,615 | 0.01% |
| CSR/系統 | 4 | 0.00% |
| 合計 | 4,360,277,319 | 100% |

最常執行的 10 種指令：

| 指令 | 類別 | 次數 | 佔比 |
|---|---|---:|---:|
| `addw` | 整數 ALU | 1,303,622,534 | 29.90% |
| `mulw` | 乘法 | 1,257,947,894 | 28.85% |
| `lb` | load | 972,514,770 | 22.30% |
| `addi` | 整數 ALU | 271,194,920 | 6.22% |
| `add` | 整數 ALU | 71,685,177 | 1.64% |
| `bne` | 條件分支 | 53,117,132 | 1.22% |
| `mv` | 整數 ALU | 35,244,025 | 0.81% |
| `bge` | 條件分支 | 31,928,596 | 0.73% |
| `and` | 整數 ALU | 31,319,984 | 0.72% |
| `sext.w` | 整數 ALU | 28,296,445 | 0.65% |

## 3. 記憶體指令

| 指令 | 寬度 | 次數 | 佔全部 load/store |
|---|---:|---:|---:|
| `lb` | 1 B | 972,514,770 | 94.96% of load |
| `lbu` | 1 B | 1,832,084 | 0.18% of load |
| `lhu` | 2 B | 7 | 0.00% of load |
| `lw` | 4 B | 22,367,523 | 2.18% of load |
| `lwu` | 4 B | 4 | 0.00% of load |
| `ld` | 8 B | 27,410,583 | 2.68% of load |
| `sb` | 1 B | 20,176,665 | 70.62% of store |
| `sw` | 4 B | 41,389 | 0.14% of store |
| `sd` | 8 B | 8,351,205 | 29.23% of store |

- load：1,024,124,971 次，共 1,283,101,640 bytes，平均每次只讀 **1.25 bytes**（暫存器寬 8 bytes）
- store：28,569,259 次，共 87,151,861 bytes，平均每次 3.05 bytes

## 4. 各運算的貢獻（make profile）

| 運算 | 次數 | 指令數 | 佔比 | MAC 數 | MAC/指令 | 輸出元素 | 指令/輸出元素 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULLY_CONNECTED | 49 | 3,225,669,711 | 73.98% | 1,045,949,952 | 0.324 | 4,085,992 | 789.4 |
| BATCH_MATMUL | 24 | 630,737,605 | 14.47% | 178,831,872 | 0.284 | 1,851,012 | 340.8 |
| MUL | 87 | 169,718,702 | 3.89% | 0 | 0.000 | 4,233,924 | 40.1 |
| CONV_2D | 1 | 88,915,787 | 2.04% | 28,901,376 | 0.325 | 37,632 | 2362.8 |
| ADD | 75 | 79,991,585 | 1.83% | 0 | 0.000 | 1,896,125 | 42.2 |
| SOFTMAX | 12 | 75,093,356 | 1.72% | 0 | 0.000 | 1,397,124 | 53.7 |
| SUB | 25 | 39,840,456 | 0.91% | 0 | 0.000 | 945,600 | 42.1 |
| GELU | 12 | 16,342,704 | 0.37% | 0 | 0.000 | 1,815,552 | 9.0 |
| TRANSPOSE | 27 | 14,076,519 | 0.32% | 0 | 0.000 | 2,041,344 | 6.9 |
| MEAN | 50 | 7,951,885 | 0.18% | 0 | 0.000 | 9,850 | 807.3 |
| SLICE | 36 | 7,006,511 | 0.16% | 0 | 0.000 | 1,361,664 | 5.1 |
| RSQRT | 25 | 1,969,248 | 0.05% | 0 | 0.000 | 4,925 | 399.8 |
| QUANTIZE | 2 | 1,949,720 | 0.04% | 0 | 0.000 | 37,824 | 51.5 |
| RESHAPE | 61 | 995,764 | 0.02% | 0 | 0.000 | 3,214,848 | 0.3 |
| CONCATENATION | 1 | 12,457 | 0.00% | 0 | 0.000 | 37,824 | 0.3 |
| STRIDED_SLICE | 1 | 977 | 0.00% | 0 | 0.000 | 192 | 5.1 |
| BROADCAST_TO | 1 | 718 | 0.00% | 0 | 0.000 | 192 | 3.7 |

## 5. 各類程式碼的指令組成（依函式名稱歸類）

| 歸類 | 指令數 | 佔比 | load | store | ALU | 乘法 | 分支 | 跳躍 | 其他 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 其他（框架/控制） | 3,858,799,694 | 88.50% | 25.00% | 0.36% | 41.15% | 31.90% | 1.59% | 0.00% | 0.00% |
| MUL | 171,399,777 | 3.93% | 5.07% | 2.48% | 74.94% | 4.97% | 12.50% | 0.04% | 0.00% |
| ADD/SUB | 119,800,991 | 2.75% | 9.59% | 2.37% | 73.77% | 2.37% | 11.88% | 0.02% | 0.00% |
| CONV_2D | 88,742,862 | 2.04% | 24.70% | 0.09% | 41.30% | 32.61% | 1.29% | 0.00% | 0.00% |
| SOFTMAX | 74,822,485 | 1.72% | 9.36% | 1.90% | 69.83% | 1.92% | 16.94% | 0.05% | 0.00% |
| TRANSPOSE/SLICE/搬資料 | 22,256,004 | 0.51% | 21.49% | 19.24% | 41.46% | 0.00% | 17.03% | 0.78% | 0.00% |
| GELU | 16,340,544 | 0.37% | 22.22% | 11.11% | 55.56% | 0.00% | 11.11% | 0.00% | 0.00% |
| MEAN | 7,942,185 | 0.18% | 23.95% | 0.13% | 51.35% | 0.12% | 24.44% | 0.00% | 0.00% |
| RSQRT | 172,777 | 0.00% | 28.69% | 8.65% | 39.71% | 0.00% | 14.31% | 8.63% | 0.00% |

## 6. 最熱的三個迴圈（每一圈的組成）

**1. `{anon}::FcB1Eval`** — 95 條/圈 × 32,514,048 圈 = 3,088,834,560 條（70.84%）

每圈：load 24、整數 ALU 38、乘法 32、條件分支 1

```
lb     s10,0(t0)
lb     s2,0(a0)
lb     t6,0(t2)
lb     s8,0(t3)
mulw   s3,s10,s2
lb     s7,0(t4)
lb     s4,0(t5)
addi   t0,t0,4
addi   t2,t2,4
addi   t5,t5,4
addi   a0,a0,4
addi   t3,t3,4
addi   t4,t4,4
mulw   s6,s10,s4
addw   s3,s3,a7
mulw   s4,t6,s4
addw   s6,s6,t1
lb     t1,-3(t3)
mulw   a7,t6,s8
addw   s4,s4,a1
lb     a1,-3(t2)
mulw   s2,t6,s2
addw   a7,a7,a4
lb     a4,-3(t0)
mulw   t6,t6,s7
addw   s2,s2,a3
lb     a3,-3(t5)
addw   a5,t6,a5
lb     t6,-3(a0)
mulw   s5,s10,s8
mulw   s8,a4,t6
addw   s5,s5,a6
mulw   s10,s10,s7
lb     s7,-3(t4)
addw   s8,s8,s3
mulw   a6,a4,a3
addw   a2,s10,a2
mulw   s3,a4,t1
addw   a6,a6,s6
lb     s6,-2(t4)
mulw   a3,a1,a3
addw   s3,s3,s5
lb     s5,-2(a0)
mulw   t6,a1,t6
addw   a3,a3,s4
lb     s4,-2(t3)
mulw   t1,a1,t1
addw   t6,t6,s2
lb     s2,-2(t2)
mulw   a1,a1,s7
addw   t1,t1,a7
addw   a1,a1,a5
mulw   a4,a4,s7
lb     a5,-2(t0)
lb     s7,-2(t5)
mulw   a7,a5,s7
addw   a4,a4,a2
addw   a7,a7,a6
mulw   a2,a5,s4
mulw   a6,a5,s5
addw   a2,a2,s3
mulw   s7,s2,s7
addw   a6,a6,s8
mulw   s5,s2,s5
addw   s7,s7,a3
lb     a3,-1(a0)
mulw   s4,s2,s4
addw   s5,s5,t6
lb     t6,-1(t0)
mulw   s2,s2,s6
addw   s4,s4,t1
addw   s2,s2,a1
lb     a1,-1(t5)
mulw   a5,a5,s6
lb     s6,-1(t4)
mulw   t1,t6,a1
addw   s3,a5,a4
lb     a5,-1(t2)
lb     a4,-1(t3)
addw   t1,t1,a7
mulw   a7,t6,a3
mulw   a1,a5,a1
addw   a7,a7,a6
mulw   a3,a5,a3
addw   a1,a1,s7
mulw   a6,t6,a4
addw   a3,a3,s5
mulw   a4,a5,a4
addw   a6,a6,a2
mulw   t6,t6,s6
addw   a4,a4,s4
mulw   a5,a5,s6
addw   a2,t6,s3
addw   a5,a5,s2
bne    t0,s1,8001e40e
```

**2. `{anon}::BmmB1Eval`** — 95 條/圈 × 5,531,904 圈 = 525,530,880 條（12.05%）

每圈：load 24、整數 ALU 38、乘法 32、條件分支 1

```
lb     s10,0(t2)
lb     t0,0(t5)
lb     t6,0(s0)
lb     s9,0(t1)
mulw   s1,s10,t0
lb     s8,0(t3)
lb     s3,0(t4)
addi   t2,t2,4
addi   s0,s0,4
addi   t4,t4,4
addi   t5,t5,4
addi   t1,t1,4
addi   t3,t3,4
mulw   s6,s10,s3
addw   s1,s1,a6
mulw   s3,t6,s3
addw   s6,s6,a7
lb     a7,-3(t1)
mulw   a6,t6,s9
addw   s3,s3,a1
lb     a1,-3(s0)
mulw   t0,t6,t0
addw   a6,a6,a4
lb     a4,-3(t2)
mulw   t6,t6,s8
addw   t0,t0,a3
lb     a3,-3(t4)
addw   a5,t6,a5
lb     t6,-3(t5)
mulw   s4,s10,s9
mulw   s9,a4,t6
addw   s4,s4,a0
mulw   s10,s10,s8
lb     s8,-3(t3)
addw   s9,s9,s1
mulw   a0,a4,a3
addw   a2,s10,a2
mulw   s1,a4,a7
addw   a0,a0,s6
lb     s6,-2(t3)
mulw   a3,a1,a3
addw   s1,s1,s4
lb     s4,-2(t5)
mulw   t6,a1,t6
addw   a3,a3,s3
lb     s3,-2(t1)
mulw   a7,a1,a7
addw   t6,t6,t0
lb     t0,-2(s0)
mulw   a1,a1,s8
addw   a7,a7,a6
addw   a1,a1,a5
mulw   a4,a4,s8
lb     a5,-2(t2)
lb     s8,-2(t4)
mulw   a6,a5,s8
addw   a4,a4,a2
addw   a6,a6,a0
mulw   a2,a5,s3
mulw   a0,a5,s4
addw   a2,a2,s1
mulw   s8,t0,s8
addw   a0,a0,s9
mulw   s4,t0,s4
addw   s8,s8,a3
lb     a3,-1(t5)
mulw   s3,t0,s3
addw   s4,s4,t6
lb     t6,-1(t2)
mulw   t0,t0,s6
addw   s3,s3,a7
addw   t0,t0,a1
lb     a1,-1(t4)
mulw   a5,a5,s6
lb     s6,-1(t3)
mulw   a7,t6,a1
addw   s1,a5,a4
lb     a5,-1(s0)
lb     a4,-1(t1)
addw   a7,a7,a6
mulw   a6,t6,a3
mulw   a1,a5,a1
addw   a6,a6,a0
mulw   a3,a5,a3
addw   a1,a1,s8
mulw   a0,t6,a4
addw   a3,a3,s4
mulw   a4,a5,a4
addw   a0,a0,a2
mulw   t6,t6,s6
addw   a4,a4,s3
mulw   a5,a5,s6
addw   a2,t6,s1
addw   a5,a5,t0
bne    t2,s2,8001ae76
```

**3. `{anon}::ConvB1Eval`** — 96 條/圈 × 903,168 圈 = 86,704,128 條（1.99%）

每圈：load 24、整數 ALU 39、乘法 32、條件分支 1

```
lb     s11,0(t0)
lb     s3,0(t3)
lb     t5,0(t2)
lb     s9,0(t6)
mulw   s4,s11,s3
lb     s8,0(t4)
lb     s5,0(t1)
addi   t6,t6,4
addi   t0,t0,4
addi   t2,t2,4
addi   t1,t1,4
addi   t3,t3,4
addi   t4,t4,4
mulw   s7,s11,s5
addw   s4,s4,a6
mulw   s5,t5,s5
addw   s7,s7,a7
lb     a7,-3(t6)
mulw   a6,t5,s9
addw   s5,s5,a1
lb     a1,-3(t2)
mulw   s3,t5,s3
addw   a6,a6,a4
lb     a4,-3(t0)
mulw   t5,t5,s8
addw   s3,s3,a3
lb     a3,-3(t1)
addw   a5,t5,a5
lb     t5,-3(t3)
mulw   s6,s11,s9
mulw   s9,a4,t5
addw   s6,s6,a0
mulw   s11,s11,s8
lb     s8,-3(t4)
addw   s9,s9,s4
mulw   a0,a4,a3
addw   a2,s11,a2
mulw   s4,a4,a7
addw   a0,a0,s7
lb     s7,-2(t4)
mulw   a3,a1,a3
addw   s4,s4,s6
lb     s6,-2(t3)
mulw   t5,a1,t5
addw   a3,a3,s5
lb     s5,-2(t6)
mulw   a7,a1,a7
addw   t5,t5,s3
lb     s3,-2(t2)
mulw   a1,a1,s8
addw   a7,a7,a6
addw   a1,a1,a5
mulw   a4,a4,s8
lb     a5,-2(t0)
lb     s8,-2(t1)
mulw   a6,a5,s8
addw   a4,a4,a2
addw   a6,a6,a0
mulw   a2,a5,s5
mulw   a0,a5,s6
addw   a2,a2,s4
mulw   s8,s3,s8
addw   a0,a0,s9
mulw   s6,s3,s6
addw   s8,s8,a3
lb     a3,-1(t3)
mulw   s5,s3,s5
addw   s6,s6,t5
lb     t5,-1(t0)
mulw   s3,s3,s7
addw   s5,s5,a7
addw   s3,s3,a1
lb     a1,-1(t1)
mulw   a5,a5,s7
lb     s7,-1(t4)
mulw   a7,t5,a1
addw   s4,a5,a4
lb     a5,-1(t2)
lb     a4,-1(t6)
addw   a7,a7,a6
mulw   a6,t5,a3
mulw   a1,a5,a1
addw   a6,a6,a0
mulw   a3,a5,a3
addw   a1,a1,s8
mulw   a0,t5,a4
addw   a3,a3,s6
mulw   a4,a5,a4
addw   a0,a0,a2
mulw   t5,t5,s7
addw   a4,a4,s5
mv     s5,s1
mulw   a5,a5,s7
addw   a2,t5,s4
addw   a5,a5,s3
bne    t6,s1,8001c2c0
```

## 7. Code size

| 區段 | bytes | 內容 |
|---|---:|---|
| .text | 194,744 | 程式碼（LiteRT Micro + kernel + 驅動） |
| .rodata | 6,807,456 | 唯讀資料：內嵌的 INT8 模型、輸入圖片、離線參數表 |
| .data | 2,944 | 有初始值的全域變數 |
| .bss | 8,392,088 | 未初始化的全域變數，主要是 8 MiB tensor arena |

主要 kernel 函式大小：

| 函式 | 運算 | bytes |
|---|---|---:|
| `tflite::micro::{anon}::FullyConnectedEval` | FC | 3,108 |
| `tflite::micro::{anon}::BatchMatMulEval` | BATCH_MATMUL | 5,188 |
| `tflite::micro::reference_integer_ops::ConvPerChannel` | CONV_2D | 768 |
| `tflite::micro::FixedPoint<int, 0> tflite::micro::exp_on_negative_values<int, 5>` | SOFTMAX exp | 722 |
| `tflite::micro::MultiplyByQuantizedMultiplier` | requant | 140 |

## 8. 量測摘要（由本次數據自動整理）

- 最熱的兩個迴圈（`{anon}::FcB1Eval`、`{anon}::BmmB1Eval`）合計佔推論的 82.89%
- 整個模型平均每個 MAC 3.48 條指令（MAC/指令 = 0.288）
- load 佔 23.49%；每個 MAC 0.82 條 load；平均每次 load 1.25 bytes；`lb`/`lbu` 佔 load 的 95.14%
- 乘法指令佔 29.19%；每個 MAC 1.02 條乘法
- 各運算每個 MAC 的指令數：CONV_2D 3.08、FULLY_CONNECTED 3.08、BATCH_MATMUL 3.53
- 各運算每個輸出元素的指令數：ADD 42.2、SUB 42.1、MUL 40.1、MEAN 807.3、SOFTMAX 53.7、GELU 9.0

