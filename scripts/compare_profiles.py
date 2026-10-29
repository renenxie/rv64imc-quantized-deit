#!/usr/bin/env python3
"""比較多個版本（B0 / B1 / X1 / X2…）的各運算指令數，輸出 Markdown 表格。只用 Python 標準函式庫。

    python3 scripts/compare_profiles.py B0=results/runtime_b0_profile_ops.csv \\
        B1=results/runtime_b1_profile_ops.csv -o results/compare_b0_b1.md

輸入：make profile 產生的 *_profile_ops.csv（每個運算一行：op_index, op, instructions, percent）
輸出：各類運算在每個版本的指令數、相對第一個版本的倍數，以及整體加速
"""
import argparse
import collections


def load(path):
    per = collections.OrderedDict()
    for ln in open(path, encoding="utf-8"):
        p = ln.strip().split(",")
        if not p[0].isdigit():
            continue
        per[p[1]] = per.get(p[1], 0) + int(p[2])
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("versions", nargs="+", help="名稱=CSV 路徑")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    names, data = [], []
    for v in a.versions:
        n, path = v.split("=", 1)
        names.append(n)
        data.append(load(path))
    base = data[0]
    ops = sorted(base, key=lambda k: -base[k])
    for d in data[1:]:
        ops += [k for k in d if k not in ops]

    head = "| 運算 | " + " | ".join(f"{n} 指令數" for n in names) + " | " + \
           " | ".join(f"{names[0]}→{n}" for n in names[1:]) + " |"
    sep = "|---|" + "---:|" * (len(names) + len(names) - 1)
    lines = [f"# 各版本指令數比較（{' / '.join(names)}）", "", head, sep]
    totals = [sum(d.values()) for d in data]
    for op in ops:
        vals = [d.get(op, 0) for d in data]
        cells = [f"{v:,}" for v in vals]
        ratios = [f"{vals[0] / v:.2f}×" if v else "-" for v in vals[1:]]
        lines.append(f"| {op} | " + " | ".join(cells) + " | " + " | ".join(ratios) + " |")
    lines.append("| **合計** | " + " | ".join(f"**{t:,}**" for t in totals) + " | " +
                 " | ".join(f"**{totals[0] / t:.2f}×**" for t in totals[1:]) + " |")
    lines.append("")
    for n, d, t in zip(names, data, totals):
        top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
        lines.append(f"- {n} 前三大：" + "、".join(f"{k} {100 * v / t:.1f}%" for k, v in top))
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
        print(f"已輸出 {a.out}")


if __name__ == "__main__":
    main()
