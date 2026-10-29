#!/usr/bin/env python3
"""從 Spike 的 -l trace 統計 instruction mix（只算指定函式的位址範圍）。

用法（從 stdin 讀 trace，可處理很大的串流）：
    spike --isa=rv64imac -l pk prog.elf 2>&1 >/dev/null | \
        python3 scripts/imix.py --elf prog.elf --func qfc_b0

分類：load / store / mul / branch / jump / alu / csr / other
之後加了自訂指令，把助記符加進 CUSTOM 即可單獨統計。
"""
import argparse
import collections
import re
import subprocess
import sys

LOAD = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld", "c.lw", "c.ld", "c.lwsp", "c.ldsp"}
STORE = {"sb", "sh", "sw", "sd", "c.sw", "c.sd", "c.swsp", "c.sdsp"}
MUL = {"mul", "mulh", "mulhu", "mulhsu", "mulw", "div", "divu", "divw", "divuw",
       "rem", "remu", "remw", "remuw"}
BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "blez", "bgez",
          "bltz", "bgtz", "bgt", "ble", "bgtu", "bleu", "c.beqz", "c.bnez"}
JUMP = {"jal", "jalr", "j", "jr", "ret", "call", "tail", "c.j", "c.jr", "c.jal", "c.jalr"}
CSR = {"csrr", "csrw", "csrs", "csrc", "csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi",
       "csrrci", "rdinstret", "rdcycle", "rdtime"}
CUSTOM = set()  # 例如 {"qmac8", "requant"}

TRACE_RE = re.compile(r"core\s+\d+:\s+0x([0-9a-f]+)\s+\(0x([0-9a-f]+)\)\s+(\S+)")


def classify(op):
    if op in CUSTOM:
        return "custom"
    if op in LOAD:
        return "load"
    if op in STORE:
        return "store"
    if op in MUL:
        return "mul"
    if op in BRANCH:
        return "branch"
    if op in JUMP:
        return "jump"
    if op in CSR:
        return "csr"
    return "alu"


def func_range(elf, func, nm):
    out = subprocess.run([nm, "-S", elf], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[3] == func:
            start, size = int(parts[0], 16), int(parts[1], 16)
            return start, start + size
    sys.exit(f"找不到函式 {func}（是否被 inline 了？可加 __attribute__((noinline))）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True)
    ap.add_argument("--func", required=True)
    ap.add_argument("--objdump", default="riscv-none-elf-objdump")
    ap.add_argument("--nm", default="riscv-none-elf-nm")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    lo, hi = func_range(args.elf, args.func, args.nm)
    cls = collections.Counter()
    ops = collections.Counter()
    compressed = 0
    for line in sys.stdin:
        m = TRACE_RE.match(line)
        if not m:
            continue
        pc = int(m.group(1), 16)
        if not (lo <= pc < hi):
            continue
        op = m.group(3)
        ops[op] += 1
        cls[classify(op)] += 1
        if (int(m.group(2), 16) & 0b11) != 0b11:   # 最低 2 bit 不是 11 → 16-bit 壓縮指令
            compressed += 1

    total = sum(cls.values())
    if total == 0:
        sys.exit("沒有統計到任何指令")
    print(f"函式 {args.func} [0x{lo:x}, 0x{hi:x})  動態指令數 = {total}")
    print(f"  壓縮指令比例 = {100.0*compressed/total:.1f}%")
    print("  類別:")
    for k in ["load", "store", "mul", "alu", "branch", "jump", "csr", "custom"]:
        if cls[k]:
            print(f"    {k:7s} {cls[k]:10d}  {100.0*cls[k]/total:5.1f}%")
    print(f"  最常見的 {args.top} 個指令:")
    for op, c in ops.most_common(args.top):
        print(f"    {op:10s} {c:10d}  {100.0*c/total:5.1f}%")


if __name__ == "__main__":
    main()
