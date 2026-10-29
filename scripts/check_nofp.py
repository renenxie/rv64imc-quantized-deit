#!/usr/bin/env python3
"""用 Spike 的 PC 直方圖證明「整個程式執行期間沒有任何浮點運算」。只用 Python 標準函式庫。

    spike -g --isa=rv64imc_zicsr_zicntr build/deit.elf > build/hist.txt 2>&1
    riscv-none-elf-objdump -d -C build/deit.elf > build/deit.dis
    python3 ../scripts/check_nofp.py build/hist.txt build/deit.dis

RV64IMC 沒有浮點指令；C/C++ 的 float/double 運算一定會變成呼叫軟體浮點函式（libgcc 的
__addsf3、__muldf3、__fixdfsi… 或 libm 的 expf、erfcf、sqrt…）。
本工具找出執行檔裡所有這類函式，加總它們在整個執行期間（開機到結束）被執行的指令數：
必須是 0 才 PASS。另外也檢查有沒有任何 F/D 擴充的浮點指令被執行（RV64IMC 不該有）。
這是與 fp_counter.c（--wrap 計數）互相獨立的第二種證明。
"""
import argparse
import re
import sys

# libgcc 軟體浮點：__<op><type>[n]，type = sf/df/tf（單/雙/四倍精度）
SOFTFP = re.compile(r"^__(add|sub|mul|div|neg|eq|ne|lt|le|gt|ge|unord|cmp|fix|fixuns|float|floatun|"
                    r"extend|trunc|powi)\w*(sf|df|tf|hf)\d?$")
# libm 與其內部函式
LIBM = re.compile(r"^(__ieee754_\w+|__kernel_\w+|__math_\w+|(sqrt|exp|exp2|expm1|log|log2|log10|log1p|"
                  r"pow|tanh|sinh|cosh|sin|cos|tan|atan|atan2|asin|acos|erf|erfc|round|lround|llround|"
                  r"floor|ceil|trunc|rint|lrint|nearbyint|frexp|ldexp|scalbn|modf|fmod|hypot|cbrt|fabs|"
                  r"copysign|nan)f?)$")
FP_INSN = re.compile(r"^f[a-z.]+$")  # F/D 擴充指令（flw、fadd.s、fcvt…）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hist")
    ap.add_argument("dis")
    a = ap.parse_args()

    counts = {}
    for ln in open(a.hist, encoding="utf-8", errors="replace"):
        m = re.match(r"^([0-9a-f]{8,16}) (\d+)$", ln.strip())
        if m:
            counts[int(m.group(1), 16)] = int(m.group(2))

    func_of, mn_of, cur = {}, {}, None
    fp_funcs = set()
    for ln in open(a.dis, encoding="utf-8", errors="replace"):
        s = ln.rstrip()
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", s)
        if m:
            cur = m.group(1)
            base = re.sub(r"\(.*", "", cur).replace("__wrap_", "").replace("__real_", "")
            if SOFTFP.match(base) or LIBM.match(base):
                fp_funcs.add(cur)
            continue
        m = re.match(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)", s)
        if m and cur:
            pc = int(m.group(1), 16)
            func_of[pc] = cur
            mn_of[pc] = m.group(2)

    total = sum(counts.values())
    fp_exec = {}
    fp_insn = 0
    for pc, n in counts.items():
        f = func_of.get(pc)
        if f in fp_funcs:
            fp_exec[f] = fp_exec.get(f, 0) + n
        if FP_INSN.match(mn_of.get(pc, "")) and mn_of.get(pc) not in ("fence", "fence.i"):
            fp_insn += n

    fp_total = sum(fp_exec.values())
    print(f"整個程式執行 {total:,} 條指令")
    print(f"執行檔中的軟體浮點/數學函式：{len(fp_funcs)} 個（只是連結進來；重點是有沒有被執行）")
    print(f"這些函式被執行的指令數：{fp_total:,}")
    for f, n in sorted(fp_exec.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {n:>14,}  {f}")
    print(f"F/D 浮點指令被執行：{fp_insn:,}")
    ok = fp_total == 0 and fp_insn == 0
    print("NOFP " + ("PASS：整個執行期間沒有任何浮點運算" if ok else "FAIL：仍有浮點運算"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
