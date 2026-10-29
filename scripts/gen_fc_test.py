#!/usr/bin/env python3
"""產生 INT8 FC 測試資料與「軟體參考答案」(golden)，輸出成 C header。

參考答案用純 Python 以相同的 TFLite 整數語意計算，與 RISC-V 上的 C kernel 完全獨立，
用來驗證 kernel（以及之後的 Xqmac / requant 指令）逐位元正確。

用法：
    python3 scripts/gen_fc_test.py --M 4 --K 40 --N 12 --seed 1 -o build/fc_small.h
"""
import argparse
import math
import random

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


def quantize_multiplier(real):
    """real ≈ qm * 2^(shift-31)，qm ∈ [2^30, 2^31)。對應 TFLite QuantizeMultiplier。"""
    if real == 0.0:
        return 0, 0
    q, shift = math.frexp(real)          # real = q * 2^shift, q ∈ [0.5, 1)
    qm = round(q * (1 << 31))
    if qm == (1 << 31):
        qm //= 2
        shift += 1
    return qm, shift


def trunc_div(a, b):
    """C 語言的整數除法（向 0 截斷）。"""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def srdhm(a, b):
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    return trunc_div(ab + nudge, 1 << 31)


def rdbpot(x, e):
    mask = (1 << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> e) + (1 if rem > thr else 0)


def mbqm(x, qm, shift):
    ls = shift if shift > 0 else 0
    rs = 0 if shift > 0 else -shift
    xs = ((x << ls) + (1 << 31)) % (1 << 32) - (1 << 31)  # 模擬 C 的 32-bit 截斷
    return rdbpot(srdhm(xs, qm), rs)


def c_array(ctype, name, vals, per_line=16):
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("  " + ", ".join(str(v) for v in vals[i:i + per_line]) + ",")
    return f"static const {ctype} {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--relu", action="store_true", help="輸出加 ReLU（act_min = out_zp）")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    M, K, N = args.M, args.K, args.N
    rng = random.Random(args.seed)

    in_zp = rng.randint(-20, 20)
    out_zp = rng.randint(-10, 10)
    act_min = out_zp if args.relu else -128
    act_max = 127

    inp = [rng.randint(-128, 127) for _ in range(M * K)]
    w = [rng.randint(-127, 127) for _ in range(N * K)]   # 對稱量化，不用 -128
    bias = [rng.randint(-20000, 20000) for _ in range(N)]

    # 讓輸出大致落在 int8 範圍內：acc 標準差 ≈ sqrt(K) * 74 * 73
    acc_std = math.sqrt(K) * 74.0 * 73.0
    qm, sh = [], []
    for _ in range(N):
        real = (40.0 / acc_std) * rng.uniform(0.5, 1.5)
        a, b = quantize_multiplier(real)
        qm.append(a)
        sh.append(b)

    out = []
    for m in range(M):
        for n in range(N):
            acc = bias[n]
            for k in range(K):
                acc += (inp[m * K + k] - in_zp) * w[n * K + k]
            v = mbqm(acc, qm[n], sh[n]) + out_zp
            out.append(max(act_min, min(act_max, v)))

    sat = sum(1 for v in out if v in (-128, 127))
    with open(args.out, "w") as f:
        f.write(f"// 自動產生：gen_fc_test.py --M {M} --K {K} --N {N} --seed {args.seed}"
                f"{' --relu' if args.relu else ''}\n// 請勿手動修改\n#pragma once\n#include <stdint.h>\n\n")
        f.write(f"#define T_M {M}\n#define T_K {K}\n#define T_N {N}\n")
        f.write(f"#define T_IN_ZP {in_zp}\n#define T_OUT_ZP {out_zp}\n")
        f.write(f"#define T_ACT_MIN {act_min}\n#define T_ACT_MAX {act_max}\n\n")
        f.write(c_array("int8_t", "t_in", inp))
        f.write(c_array("int8_t", "t_w", w))
        f.write(c_array("int32_t", "t_bias", bias))
        f.write(c_array("int32_t", "t_qm", qm))
        f.write(c_array("int8_t", "t_shift", sh))
        f.write(c_array("int8_t", "t_golden", out))
    print(f"[gen] {args.out}: M={M} K={K} N={N}, MACs={M*K*N}, "
          f"飽和輸出 {sat}/{len(out)} ({100.0*sat/len(out):.1f}%)")


if __name__ == "__main__":
    main()
