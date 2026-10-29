# Spike 原始碼（期中 project 用）

這是 RISC-V 官方指令集模擬器 Spike 的原始碼，放進專案是為了加入自訂指令（packed INT8 MAC、requant）。

| 項目 | 內容 |
|---|---|
| 來源 | https://github.com/riscv-software-src/riscv-isa-sim |
| 版本 | commit `7ab2efd6785e847c6d13d810c0b25b7c9c26bd21`（2026-09-25） |
| 取得方式 | `git archive` 該 commit，原封不動放在這裡 |
| 目前狀態 | **尚未修改**（與官方原始碼完全相同） |
| 唯一差異 | 官方的 `spike_dasm/spike_dasm_option_parser.cc` 是指向 `../fesvr/option_parser.cc` 的符號連結；Mac 的掛載資料夾不支援，改成內容相同的一般檔案 |
| 授權 | 見 `LICENSE`（BSD） |

與 `scripts/setup_workstation.sh` 固定的版本相同；B0 的所有量測（`invoke_insts=12617761175`）都是用這個版本得到的。

## 編譯與使用（在 rv64-dev 容器內）

```bash
bash /work/scripts/build_spike.sh          # 第一次約 5～10 分鐘；之後只重編改到的部分
/work/spike/build/spike --isa=rv64imc_zicsr_zicntr <程式.elf>
make -C /work/runtime run SPIKE=/work/spike/build/spike   # 用這份 Spike 跑完整模型
```

編譯產物在 `spike/build/`（不進 git）。

## 之後加新指令會改的地方

| 檔案 | 內容 |
|---|---|
| `riscv/encoding.h` | 新指令的編碼（MATCH / MASK） |
| `riscv/insns/<指令>.h` | 執行語意 |
| `riscv/riscv.mk.in` | 把指令加進編譯清單 |
| `disasm/disasm.cc` | 反組譯（trace 顯示指令名稱） |
| `riscv/isa_parser.cc` | 擴充名稱，例如 `--isa=rv64imc_xqmac` |

修改處一律標註 `[期中project 修改]`，方便和官方原始碼對照。
