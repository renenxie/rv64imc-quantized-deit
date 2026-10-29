#!/usr/bin/env python3
"""找出最熱的迴圈：讀 Spike 的 PC 直方圖（spike -g），配上反組譯，列出最耗指令的程式片段。

    spike -g --isa=rv64imc_zicsr_zicntr runtime/build/deit.elf > runtime/build/hist.txt 2>&1
    riscv-none-elf-objdump -d -C runtime/build/deit.elf > runtime/build/deit.dis
    python3 scripts/hot_loops.py runtime/build/hist.txt runtime/build/deit.dis -o results/hot_loops_b0.md

做法：相鄰、且執行次數相同的指令視為同一段（通常就是一個迴圈本體），
      依「執行次數 × 指令數」排序，印出每段的反組譯與佔整體的比例。
（只用 Python 標準函式庫）
"""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hist")
    ap.add_argument("dis")
    ap.add_argument("-n", "--top", type=int, default=6)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    counts, total_dyn = {}, None
    for ln in open(a.hist, encoding="utf-8", errors="replace"):
        m = re.match(r"^([0-9a-f]{8,16}) (\d+)$", ln.strip())
        if m:
            counts[int(m.group(1), 16)] = int(m.group(2))
        m = re.search(r"invoke_insts=(\d+)", ln)
        if m:
            total_dyn = int(m.group(1))
    all_dyn = sum(counts.values())

    # 反組譯：位址 → (指令文字, 所在函式)
    insn, func, cur = {}, {}, "?"
    for ln in open(a.dis, encoding="utf-8", errors="replace"):
        m = re.match(r"^([0-9a-f]+) <(.*)>:$", ln.strip())
        if m:
            cur = m.group(2)
            continue
        m = re.match(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(.*)$", ln.rstrip())
        if m:
            pc = int(m.group(1), 16)
            insn[pc] = re.sub(r"\s+<.*>$", "", m.group(2)).replace("\t", " ")
            func[pc] = cur

    pcs = sorted(counts)
    blocks, i = [], 0
    while i < len(pcs):
        j = i
        while j + 1 < len(pcs) and counts[pcs[j + 1]] == counts[pcs[i]] and pcs[j + 1] - pcs[j] <= 4:
            j += 1
        blk = pcs[i:j + 1]
        blocks.append((counts[blk[0]] * len(blk), blk))
        i = j + 1
    blocks.sort(key=lambda b: -b[0])

    base = total_dyn or all_dyn
    out = [f"# 最熱的程式片段（Spike PC 直方圖）", "",
           f"整個程式執行 {all_dyn:,} 條指令；推論（Invoke）{base:,} 條。比例以推論為分母。", ""]
    for rank, (dyn, blk) in enumerate(blocks[:a.top], 1):
        f = func.get(blk[0], "?").replace("(anonymous namespace)", "{anon}")
        f = re.sub(r"\(.*", "", f)
        out += [f"## {rank}. `{f}`",
                f"- 每段 {len(blk)} 條指令 × 執行 {counts[blk[0]]:,} 次 = {dyn:,} 條（{100*dyn/base:.2f}%）", "",
                "```"]
        out += [f"{pc:08x}:  {insn.get(pc, '?')}" for pc in blk]
        out += ["```", ""]
    text = "\n".join(out)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"已輸出 {a.out}")


if __name__ == "__main__":
    main()
