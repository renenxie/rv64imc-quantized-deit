#!/usr/bin/env python3
"""Workload 分析（規格 2.B）：從 Spike 的 PC 直方圖算出推論的指令組成、記憶體指令、MAC/指令、
各運算貢獻、code size，並整理出前幾大瓶頸。只用 Python 標準函式庫。

    （在 runtime/ 下）make analysis        # 會自動跑完下面全部

    手動：
    spike -g --isa=rv64imc_zicsr_zicntr build/deit.elf          > build/hist.txt 2>&1
    spike -g --isa=rv64imc_zicsr_zicntr build/deit_noinvoke.elf > build/hist_noinvoke.txt 2>&1
    riscv-none-elf-objdump -d -C build/deit.elf > build/deit.dis
    riscv-none-elf-size -A build/deit.elf > build/size.txt
    python3 ../scripts/workload_analysis.py --hist build/hist.txt --hist-base build/hist_noinvoke.txt \\
        --dis build/deit.dis --size build/size.txt --macs ../results/model_macs.txt \\
        --profile ../results/runtime_b0_profile_ops.csv -o ../results/workload_b0

方法：
  - deit.elf 與 deit_noinvoke.elf 的程式碼位址完全相同，差別只在後者跳過 Invoke；
    兩份直方圖相減 = 只屬於推論（Invoke）的每個 PC 執行次數。
  - 每個 PC 依反組譯的助記符分類（load / store / ALU / mul / div / branch / jump / system）。
  - MAC 數來自 scripts/count_macs.py（由模型形狀計算），各運算指令數來自 make profile。
"""
import argparse
import collections
import re

# ---------------------------------------------------------------- 指令分類（RV64IMC + Zicsr）
LOAD_W = {"lb": 1, "lbu": 1, "lh": 2, "lhu": 2, "lw": 4, "lwu": 4, "ld": 8}
STORE_W = {"sb": 1, "sh": 2, "sw": 4, "sd": 8}
MUL = {"mul", "mulh", "mulhsu", "mulhu", "mulw"}
DIV = {"div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw"}
BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "blez", "bgez", "bltz", "bgtz",
          "bgt", "ble", "bgtu", "bleu"}
JUMP = {"jal", "j", "jalr", "jr", "ret", "call", "tail"}
SYSTEM = {"csrr", "csrw", "csrs", "csrc", "csrrw", "csrrs", "csrrc", "csrwi", "csrsi", "csrci",
          "rdinstret", "rdcycle", "rdtime", "fence", "fence.i", "ecall", "ebreak", "wfi", "mret", "nop"}
CLASSES = ["load", "store", "alu", "mul", "div", "branch", "jump", "system"]
CLASS_ZH = {"load": "load", "store": "store", "alu": "整數 ALU", "mul": "乘法", "div": "除法/餘數",
            "branch": "條件分支", "jump": "跳躍/呼叫", "system": "CSR/系統"}


def classify(mn):
    if mn in LOAD_W:
        return "load"
    if mn in STORE_W:
        return "store"
    if mn in MUL:
        return "mul"
    if mn in DIV:
        return "div"
    if mn in BRANCH:
        return "branch"
    if mn in JUMP:
        return "jump"
    if mn in SYSTEM or mn.startswith("csr"):
        return "system"
    return "alu"


# 函式名稱 → 所屬運算（依函式歸類；共用函式另列）
GROUPS = [
    ("FULLY_CONNECTED", r"FullyConnected"),
    ("BATCH_MATMUL", r"BatchMatMul"),
    ("CONV_2D", r"Conv"),
    ("SOFTMAX", r"Softmax|exp_on_negative|one_over_one_plus|GetReciprocal|FixedPoint|Rescale|"
                r"SaturatingRoundingMultiplyByPOT|RoundingDivideByPOT|exp_on_interval"),
    ("MEAN", r"Mean|Reduce|QuantizedMeanOrSum"),
    ("ADD/SUB", r"\bAdd|Sub[A-Z(]|AddEval|SubEval|BroadcastAdd|BroadcastSub|ElementWise"),
    ("MUL", r"\bMul|MulEval|BroadcastMul"),
    ("RSQRT", r"Rsqrt"),
    ("GELU", r"Gelu"),
    ("TRANSPOSE/SLICE/搬資料", r"Transpose|Slice|Concat|Reshape|BroadcastTo|memcpy|memmove|memset"),
    ("requant（共用）", r"MultiplyByQuantizedMultiplier"),
]


def group_of(func):
    for g, pat in GROUPS:
        if re.search(pat, func):
            return g
    return "其他（框架/控制）"


def read_hist(path):
    h, invoke = {}, None
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^([0-9a-f]{8,16}) (\d+)$", ln.strip())
        if m:
            h[int(m.group(1), 16)] = int(m.group(2))
        m = re.search(r"invoke_insts=(\d+)", ln)
        if m:
            invoke = int(m.group(1))
    return h, invoke


def read_dis(path):
    insn, func, starts, cur = {}, {}, [], "?"
    for ln in open(path, encoding="utf-8", errors="replace"):
        s = ln.rstrip()
        m = re.match(r"^([0-9a-f]+) <(.*)>:$", s)
        if m:
            cur = m.group(2)
            starts.append((int(m.group(1), 16), cur))
            continue
        m = re.match(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)\s*(.*)$", s)
        if m:
            pc = int(m.group(1), 16)
            insn[pc] = (m.group(3), len(m.group(2)) // 2, re.sub(r"\s+[#<].*$", "", m.group(4)))
            func[pc] = cur
    return insn, func, starts


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--hist-base", required=True)
    ap.add_argument("--dis", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--macs", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--label", default="B0（LiteRT Micro reference kernel，RV64IMC，-O2）",
                    help="報告標題中的版本說明")
    a = ap.parse_args()

    full, invoke_insts = read_hist(a.hist)
    base, _ = read_hist(a.hist_base)
    insn, func, starts = read_dis(a.dis)

    # 只屬於推論的 PC 執行次數
    inv = {}
    for pc, n in full.items():
        d = n - base.get(pc, 0)
        if d > 0:
            inv[pc] = d
    total = sum(inv.values())

    # MAC 與輸出元素
    macs_type, macs_op, elem_op = {}, {}, {}
    total_macs = 0
    for ln in open(a.macs, encoding="utf-8"):
        p = ln.strip().split(",")
        if p[0] == "TYPE":
            macs_type[p[1]] = int(p[3])
        elif p[0] == "TOTAL_MACS":
            total_macs = int(p[1])
        elif p[0] == "OP":
            macs_op[int(p[1])] = int(p[3])
        elif p[0] == "ELEM":
            elem_op[int(p[1])] = (p[2], int(p[3]))

    # 各運算指令數（make profile）
    prof = []
    for ln in open(a.profile, encoding="utf-8"):
        p = ln.strip().split(",")
        if p[0].isdigit():
            prof.append((int(p[0]), p[1], int(p[2])))

    # ------------------------------------------------ 1. 指令組成
    mix = collections.Counter()
    by_mn = collections.Counter()
    load_w = collections.Counter()
    store_w = collections.Counter()
    gmix = collections.defaultdict(collections.Counter)
    unknown = 0
    for pc, n in inv.items():
        if pc not in insn:
            unknown += n
            continue
        mn = insn[pc][0]
        c = classify(mn)
        mix[c] += n
        by_mn[mn] += n
        if c == "load":
            load_w[mn] += n
        elif c == "store":
            store_w[mn] += n
        gmix[group_of(func[pc])][c] += n

    loads, stores = mix["load"], mix["store"]
    load_bytes = sum(LOAD_W[m] * n for m, n in load_w.items())
    store_bytes = sum(STORE_W[m] * n for m, n in store_w.items())

    L = []
    L += [f"# DeiT-Tiny INT8 workload 分析 — {a.label}", "",
          "資料來源：Spike PC 直方圖（`spike -g`），扣掉只做初始化的版本後，只留下推論（Invoke）的指令。", ""]

    L += ["## 1. 總覽", "",
          "| 項目 | 數值 |", "|---|---:|",
          f"| 推論動態指令數（直方圖相減） | {total:,} |"]
    if invoke_insts:
        L.append(f"| 推論動態指令數（程式內 instret） | {invoke_insts:,} |")
    L += [f"| 模型 MAC 數 | {total_macs:,} |",
          f"| **MAC / 指令** | **{total_macs / total:.4f}**（每個 MAC 平均 {total / total_macs:.2f} 條指令） |",
          f"| 記憶體指令（load + store） | {loads + stores:,}（{pct(loads + stores, total)}） |",
          f"| load / MAC | {loads / total_macs:.3f} |",
          f"| store / MAC | {stores / total_macs:.3f} |", ""]
    if invoke_insts:
        L += [f"兩種量法差 {abs(total - invoke_insts):,} 條（{pct(abs(total - invoke_insts), invoke_insts)}），"
              "來自 Invoke 前後讀計數器與比對程式的些微差異，可忽略。", ""]

    # ------------------------------------------------ 2. 指令組成表
    L += ["## 2. 指令組成（instruction mix）", "",
          "| 類別 | 指令數 | 佔比 |", "|---|---:|---:|"]
    for c in CLASSES:
        L.append(f"| {CLASS_ZH[c]} | {mix[c]:,} | {pct(mix[c], total)} |")
    L += [f"| 合計 | {total:,} | 100% |", ""]
    L += ["最常執行的 10 種指令：", "", "| 指令 | 類別 | 次數 | 佔比 |", "|---|---|---:|---:|"]
    for mn, n in by_mn.most_common(10):
        L.append(f"| `{mn}` | {CLASS_ZH[classify(mn)]} | {n:,} | {pct(n, total)} |")
    L.append("")

    # ------------------------------------------------ 3. 記憶體指令
    L += ["## 3. 記憶體指令", "",
          "| 指令 | 寬度 | 次數 | 佔全部 load/store |", "|---|---:|---:|---:|"]
    for m in ["lb", "lbu", "lh", "lhu", "lw", "lwu", "ld"]:
        if load_w[m]:
            L.append(f"| `{m}` | {LOAD_W[m]} B | {load_w[m]:,} | {pct(load_w[m], loads)} of load |")
    for m in ["sb", "sh", "sw", "sd"]:
        if store_w[m]:
            L.append(f"| `{m}` | {STORE_W[m]} B | {store_w[m]:,} | {pct(store_w[m], stores)} of store |")
    L += ["",
          f"- load：{loads:,} 次，共 {load_bytes:,} bytes，平均每次只讀 **{load_bytes / loads:.2f} bytes**"
          f"（暫存器寬 8 bytes）",
          f"- store：{stores:,} 次，共 {store_bytes:,} bytes，平均每次 {store_bytes / max(stores, 1):.2f} bytes", ""]

    # ------------------------------------------------ 4. 各運算貢獻
    agg = collections.OrderedDict()
    for i, name, n in prof:
        d = agg.setdefault(name, [0, 0, 0, 0])
        d[0] += 1
        d[1] += n
        d[2] += macs_op.get(i, 0)
        d[3] += elem_op.get(i, ("", 0))[1]
    ptotal = sum(n for _, _, n in prof)
    L += ["## 4. 各運算的貢獻（make profile）", "",
          "| 運算 | 次數 | 指令數 | 佔比 | MAC 數 | MAC/指令 | 輸出元素 | 指令/輸出元素 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, (c, n, mc, el) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        L.append(f"| {name} | {c} | {n:,} | {pct(n, ptotal)} | {mc:,} | "
                 f"{(mc / n if mc else 0):.3f} | {el:,} | {n / el:.1f} |")
    L.append("")

    # ------------------------------------------------ 5. 依函式歸類的指令組成
    L += ["## 5. 各類程式碼的指令組成（依函式名稱歸類）", "",
          "| 歸類 | 指令數 | 佔比 | load | store | ALU | 乘法 | 分支 | 跳躍 | 其他 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g, cnt in sorted(gmix.items(), key=lambda kv: -sum(kv[1].values())):
        s = sum(cnt.values())
        other = cnt["div"] + cnt["system"]
        L.append(f"| {g} | {s:,} | {pct(s, total)} | {pct(cnt['load'], s)} | {pct(cnt['store'], s)} | "
                 f"{pct(cnt['alu'], s)} | {pct(cnt['mul'], s)} | {pct(cnt['branch'], s)} | "
                 f"{pct(cnt['jump'], s)} | {pct(other, s)} |")
    L.append("")

    # ------------------------------------------------ 6. 最熱迴圈的組成（每一圈）
    pcs = sorted(inv)
    blocks, i = [], 0
    while i < len(pcs):
        j = i
        while j + 1 < len(pcs) and inv[pcs[j + 1]] == inv[pcs[i]] and pcs[j + 1] - pcs[j] <= 4:
            j += 1
        blk = pcs[i:j + 1]
        blocks.append((inv[blk[0]] * len(blk), blk))
        i = j + 1
    blocks.sort(key=lambda b: -b[0])
    L += ["## 6. 最熱的三個迴圈（每一圈的組成）", ""]
    hot = []
    for rank, (dyn, blk) in enumerate(blocks[:3], 1):
        f = func.get(blk[0], "?").replace("(anonymous namespace)", "{anon}")
        f = re.sub(r"\(.*", "", f)
        cnt = collections.Counter(classify(insn[p][0]) for p in blk if p in insn)
        hot.append((f, len(blk), inv[blk[0]], dyn, cnt, blk))
        desc = "、".join(f"{CLASS_ZH[c]} {cnt[c]}" for c in CLASSES if cnt[c])
        L += [f"**{rank}. `{f}`** — {len(blk)} 條/圈 × {inv[blk[0]]:,} 圈 = {dyn:,} 條（{pct(dyn, total)}）",
              "", f"每圈：{desc}", "", "```"]
        L += [f"{insn[p][0]:<6} {insn[p][2]}" for p in blk if p in insn]
        L += ["```", ""]

    # ------------------------------------------------ 7. code size
    sec = {}
    for ln in open(a.size, encoding="utf-8"):
        p = ln.split()
        if len(p) >= 3 and p[0].startswith(".") and p[1].isdigit():
            sec[p[0]] = int(p[1])
    fsize = {}
    ends = [s for s, _ in starts[1:]] + [None]
    for (s, name), e in zip(starts, ends):
        if e is not None and e > s:
            fsize[name] = e - s
    keys = [("FullyConnectedEval", "FC"), ("BatchMatMulEval", "BATCH_MATMUL"), ("ConvPerChannel", "CONV_2D"),
            ("SoftmaxInt8", "SOFTMAX"), ("exp_on_negative_values", "SOFTMAX exp"),
            ("MultiplyByQuantizedMultiplier(int, int, int)", "requant")]
    L += ["## 7. Code size", "",
          "| 區段 | bytes | 內容 |", "|---|---:|---|",
          f"| .text | {sec.get('.text', 0):,} | 程式碼（LiteRT Micro + kernel + 驅動） |",
          f"| .rodata | {sec.get('.rodata', 0):,} | 唯讀資料：內嵌的 INT8 模型、輸入圖片、離線參數表 |",
          f"| .data | {sec.get('.data', 0):,} | 有初始值的全域變數 |",
          f"| .bss | {sec.get('.bss', 0):,} | 未初始化的全域變數，主要是 8 MiB tensor arena |", "",
          "主要 kernel 函式大小：", "", "| 函式 | 運算 | bytes |", "|---|---|---:|"]
    for k, lab in keys:
        hits = [(n, s) for n, s in fsize.items() if k in n]
        for n, s in hits[:2]:
            short = re.sub(r"\(.*", "", n.replace("(anonymous namespace)", "{anon}"))
            L.append(f"| `{short}` | {lab} | {s:,} |")
    L.append("")

    # ------------------------------------------------ 8. 瓶頸
    # 第 8 節只整理「由本次數據算出的」重點，不寫固定的結論；解讀與架構建議請寫在報告裡
    def per_el(k):
        return agg[k][1] / agg[k][3] if k in agg and agg[k][3] else 0
    mac_loops = sum(h[3] for h in hot[:2])
    L += ["## 8. 量測摘要（由本次數據自動整理）", "",
          f"- 最熱的兩個迴圈（`{hot[0][0]}`、`{hot[1][0]}`）合計佔推論的 {pct(mac_loops, total)}"
          if len(hot) >= 2 else "- （迴圈不足兩個）",
          f"- 整個模型平均每個 MAC {total / total_macs:.2f} 條指令（MAC/指令 = {total_macs / total:.3f}）",
          f"- load 佔 {pct(loads, total)}；每個 MAC {loads / total_macs:.2f} 條 load；"
          f"平均每次 load {load_bytes / loads:.2f} bytes；`lb`/`lbu` 佔 load 的 "
          f"{pct(load_w['lb'] + load_w['lbu'], loads)}",
          f"- 乘法指令佔 {pct(mix['mul'], total)}；每個 MAC {mix['mul'] / total_macs:.2f} 條乘法",
          "- 各運算每個 MAC 的指令數：" + "、".join(
              f"{k} {agg[k][1] / agg[k][2]:.2f}" for k in agg if agg[k][2]),
          "- 各運算每個輸出元素的指令數：" + "、".join(
              f"{k} {per_el(k):.1f}" for k in ("ADD", "SUB", "MUL", "MEAN", "SOFTMAX", "GELU")
              if k in agg),
          ""]

    text = "\n".join(L) + "\n"
    with open(a.out + ".md", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(a.out + "_mix.csv", "w", encoding="utf-8") as fh:
        fh.write("mnemonic,class,count,percent\n")
        for mn, n in by_mn.most_common():
            fh.write(f"{mn},{classify(mn)},{n},{100 * n / total:.4f}\n")
    print(text)
    if unknown:
        print(f"（注意：{unknown:,} 次執行的 PC 不在反組譯中）")
    print(f"已輸出 {a.out}.md、{a.out}_mix.csv")


if __name__ == "__main__":
    main()
