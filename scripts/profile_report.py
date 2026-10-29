#!/usr/bin/env python3
"""把 Spike 上 LiteRT Micro profile 的輸出整理成表格（只用 Python 標準函式庫）。

    spike --isa=rv64imc_zicsr_zicntr runtime/build/deit_profile.elf > runtime/build/profile_rv.txt
    python3 scripts/profile_report.py runtime/build/profile_rv.txt -o results/runtime_b0_profile

輸入：runtime/main.cc 在 -DPROFILE 模式印出的 OP / TAG / TOTAL / RESULT 行
輸出：<prefix>.md（各類運算佔比表）、<prefix>_ops.csv（每個運算的指令數）
"""
import argparse
import collections
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-o", "--out", default=None, help="輸出檔名前綴（不含副檔名）")
    ap.add_argument("--label", default="B0（LiteRT Micro 原始 reference kernel，-O2）")
    a = ap.parse_args()

    ops, result = [], ""
    for ln in open(a.log, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if ln.startswith("OP,"):
            _, i, tag, n = ln.split(",")
            ops.append((int(i), tag, int(n)))
        elif ln.startswith("RESULT"):
            result = ln
    if not ops:
        raise SystemExit("找不到 OP 行：請用 deit_profile.elf（make profile）的輸出")

    total = sum(n for _, _, n in ops)
    per = collections.OrderedDict()
    for _, tag, n in ops:
        c, s = per.get(tag, (0, 0))
        per[tag] = (c + 1, s + n)
    rows = sorted(per.items(), key=lambda kv: -kv[1][1])

    kv = dict(x.split("=", 1) for x in result.split()[1:] if "=" in x)
    lines = [f"# DeiT-Tiny INT8 在 Spike（RV64IMC）上的動態指令數 — {a.label}", ""]
    if kv:
        lines += [f"- 一次完整推論（Invoke）：**{int(kv['invoke_insts']):,}** 條指令",
                  f"- 推論中的軟體浮點呼叫：{kv['fp_calls_in_invoke']} 次",
                  f"- 與 Python LiteRT 參考答案比對：{kv['status']}（不一致 {kv['mismatches']} / 1000）", ""]
    lines += ["| 運算 | 次數 | 指令數 | 佔比 | 累積 |", "|---|---:|---:|---:|---:|"]
    cum = 0
    for tag, (c, s) in rows:
        cum += s
        lines.append(f"| {tag} | {c} | {s:,} | {100*s/total:.2f}% | {100*cum/total:.2f}% |")
    lines.append(f"| **合計** | {len(ops)} | **{total:,}** | 100% | |")
    text = "\n".join(lines) + "\n"
    print(text)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out + ".md", "w", encoding="utf-8") as f:
            f.write(text)
        with open(a.out + "_ops.csv", "w", encoding="utf-8") as f:
            f.write("op_index,op,instructions,percent\n")
            for i, tag, n in ops:
                f.write(f"{i},{tag},{n},{100*n/total:.4f}\n")
        print(f"已輸出 {a.out}.md、{a.out}_ops.csv")


if __name__ == "__main__":
    main()
