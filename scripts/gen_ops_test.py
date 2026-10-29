#!/usr/bin/env python3
"""產生 DeiT 其他運算的測試資料與 golden（C header）。

    python3 scripts/gen_ops_test.py gelu      --n 3072            -o build/op_gelu.h
    python3 scripts/gen_ops_test.py add       --n 3072            -o build/op_add.h
    python3 scripts/gen_ops_test.py softmax   --rows 8 --L 197    -o build/op_softmax.h
    python3 scripts/gen_ops_test.py layernorm --rows 16 --C 192   -o build/op_layernorm.h
    python3 scripts/gen_ops_test.py matmul    --M 16 --K 64 --N 197 -o build/op_matmul_qk.h

每個運算有兩種結果：
  * golden（整數版）：與 C kernel 用完全相同的整數演算法 → C 必須逐位元相同
  * 浮點參考：先用浮點算、最後才量化 → 用來衡量「整數近似」本身的誤差（報告用）
"""
import argparse
import math
import random

import sys

from qref import (c_array, c_defines, choose_qparams, clamp, dequantize, err_report,
                  mbqm, quantize, quantize_multiplier, requantize, round_div)

HEADER = "// 自動產生：{cmd}\n// 請勿手動修改\n#pragma once\n#include <stdint.h>\n\n"


def gelu_f(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


# ------------------------------------------------------------ GELU（查表）
def gen_gelu(a, rng):
    s_in, zp_in = choose_qparams(-6.0, 6.0)               # FC1 輸出的典型範圍
    s_out, zp_out = choose_qparams(gelu_f(-0.7518), 6.0)   # GELU 最小值約 -0.17
    lut = [quantize(gelu_f(dequantize(q, s_in, zp_in)), s_out, zp_out) for q in range(-128, 128)]
    x = list(range(-128, 128)) + [rng.randint(-128, 127) for _ in range(a.n - 256)]
    golden = [lut[v + 128] for v in x]
    info = "查表結果 = 浮點算完再量化（誤差 0 LSB，由構造保證）"
    body = c_defines({"T_N": len(x)})
    body += c_array("int8_t", "t_in", x) + c_array("int8_t", "t_lut", lut)
    body += c_array("int8_t", "t_golden", golden)
    return body, info


# ------------------------------------------------------------ 殘差相加（TFLite ADD 語意）
def gen_add(a, rng):
    s1, zp1 = choose_qparams(-4.0, 4.5)
    s2, zp2 = choose_qparams(-2.0, 3.0)
    so, zpo = choose_qparams(-6.0, 7.5)
    lshift = 20
    twice_max = 2.0 * max(s1, s2)
    m1, sh1 = quantize_multiplier(s1 / twice_max)
    m2, sh2 = quantize_multiplier(s2 / twice_max)
    mo, sho = quantize_multiplier(twice_max / ((1 << lshift) * so))
    x1 = [rng.randint(-128, 127) for _ in range(a.n)]
    x2 = [rng.randint(-128, 127) for _ in range(a.n)]
    golden, fref = [], []
    for p, q in zip(x1, x2):
        v1 = mbqm((p - zp1) << lshift, m1, sh1)
        v2 = mbqm((q - zp2) << lshift, m2, sh2)
        golden.append(requantize(v1 + v2, mo, sho, zpo))
        fref.append(quantize(dequantize(p, s1, zp1) + dequantize(q, s2, zp2), so, zpo))
    mx, same = err_report(golden, fref)
    info = f"vs 浮點：最大誤差 {mx} LSB，完全相同 {100*same:.1f}%"
    body = c_defines({"T_N": a.n, "T_ZP1": zp1, "T_ZP2": zp2, "T_LSHIFT": lshift,
                      "T_M1": m1, "T_S1": sh1, "T_M2": m2, "T_S2": sh2,
                      "T_MO": mo, "T_SO": sho, "T_OZP": zpo})
    body += c_array("int8_t", "t_a", x1) + c_array("int8_t", "t_b", x2)
    body += c_array("int8_t", "t_golden", golden)
    return body, info


# ------------------------------------------------------------ Softmax（exp 查表 + 整數除法）
EXP_Q = 16  # exp 表的定點格式：Q16，exp(0) = 65536


def softmax_int(row, lut):
    mx = max(row)
    e = [lut[mx - v] for v in row]
    s = sum(e)
    # 輸出量化：scale = 1/256, zero point = -128（TFLite int8 softmax 慣例）
    return [clamp(((ei << 8) + s // 2) // s - 128, -128, 127) for ei in e]


def gen_softmax(a, rng):
    s_in, zp_in = choose_qparams(-10.0, 10.0)               # attention score 的範圍
    lut = [round(math.exp(-d * s_in) * (1 << EXP_Q)) for d in range(256)]
    # 每列用不同分布，讓輸出機率涵蓋大、中、小值（否則大多數輸出都是 0 → 測試太弱）
    x = []
    for r in range(a.rows):
        kind = r % 4
        if kind == 0:      # 全範圍均勻
            row = [rng.randint(-128, 127) for _ in range(a.L)]
        elif kind == 1:    # 窄範圍 → 接近均勻分布
            base = rng.randint(-100, 90)
            row = [base + rng.randint(0, 10) for _ in range(a.L)]
        elif kind == 2:    # 少數幾個特別大（像 attention 集中在少數 token）
            row = [rng.randint(-128, 0) for _ in range(a.L)]
            for _ in range(3):
                row[rng.randrange(a.L)] = rng.randint(90, 127)
        else:              # 中等範圍
            row = [rng.randint(-40, 40) for _ in range(a.L)]
        x += row
    golden, fref = [], []
    for r in range(a.rows):
        row = x[r * a.L:(r + 1) * a.L]
        golden += softmax_int(row, lut)
        f = [dequantize(v, s_in, zp_in) for v in row]
        m = max(f)
        ex = [math.exp(v - m) for v in f]
        sm = sum(ex)
        fref += [quantize(v / sm, 1.0 / 256, -128) for v in ex]
    mx, same = err_report(golden, fref)
    info = f"vs 浮點：最大誤差 {mx} LSB，完全相同 {100*same:.1f}%"
    body = c_defines({"T_ROWS": a.rows, "T_L": a.L})
    body += c_array("int8_t", "t_in", x) + c_array("uint32_t", "t_explut", lut)
    body += c_array("int8_t", "t_golden", golden)
    return body, info


# ------------------------------------------------------------ LayerNorm（整數平均/變異數/開根號）
LN_F = 16  # γ/β 的定點小數位數


def layernorm_int(row, zp, G, B, eps_int, zp_out):
    C = len(row)
    xi = [v - zp for v in row]
    s = sum(xi)
    D = [C * v - s for v in xi]                    # D = C·(x − mean)，精確整數
    V = sum(d * d for d in D)                      # = C³·var
    S = math.isqrt(V // C + eps_int) or 1          # ≈ C·std
    out = []
    for c in range(C):
        t = round_div(D[c] * G[c], S) + B[c]       # 單位：2^F × 輸出 LSB
        out.append(clamp(((t + (1 << (LN_F - 1))) >> LN_F) + zp_out, -128, 127))
    return out


def gen_layernorm(a, rng):
    C, eps = a.C, 1e-6
    mu = [rng.gauss(0.0, 1.0) for _ in range(C)]     # 每個 channel 不同的偏移（像 residual stream）
    sd = [rng.uniform(0.3, 1.5) for _ in range(C)]
    xf = [rng.gauss(mu[c], sd[c]) for _ in range(a.rows) for c in range(C)]
    s_x, zp_x = choose_qparams(min(xf), max(xf))
    x = [quantize(v, s_x, zp_x) for v in xf]
    gamma = [rng.gauss(1.0, 0.3) for _ in range(C)]
    beta = [rng.gauss(0.0, 0.3) for _ in range(C)]

    # 浮點 LayerNorm（對「已量化的輸入」做，才能單純衡量整數近似的誤差）
    yf = []
    for r in range(a.rows):
        v = [dequantize(q, s_x, zp_x) for q in x[r * C:(r + 1) * C]]
        m = sum(v) / C
        var = sum((t - m) ** 2 for t in v) / C
        yf += [(v[c] - m) / math.sqrt(var + eps) * gamma[c] + beta[c] for c in range(C)]
    s_y, zp_y = choose_qparams(min(yf), max(yf))

    G = [round(g / s_y * (1 << LN_F)) for g in gamma]
    B = [round(b / s_y * (1 << LN_F)) for b in beta]
    eps_int = round(eps * C * C / (s_x * s_x))
    golden = []
    for r in range(a.rows):
        golden += layernorm_int(x[r * C:(r + 1) * C], zp_x, G, B, eps_int, zp_y)
    fref = [quantize(v, s_y, zp_y) for v in yf]
    mx, same = err_report(golden, fref)
    info = f"vs 浮點：最大誤差 {mx} LSB，完全相同 {100*same:.1f}%"
    body = c_defines({"T_ROWS": a.rows, "T_C": C, "T_IN_ZP": zp_x, "T_OUT_ZP": zp_y,
                      "T_EPS": f"{eps_int}LL"})
    body += c_array("int8_t", "t_in", x)
    body += c_array("int32_t", "t_G", G) + c_array("int32_t", "t_B", B)
    body += c_array("int8_t", "t_golden", golden)
    return body, info


# ------------------------------------------------------------ Attention 矩陣乘法（兩邊都是 activation）
def gen_matmul(a, rng):
    M, K, N = a.M, a.K, a.N
    azp, bzp, ozp = rng.randint(-20, 20), rng.randint(-20, 20), rng.randint(-10, 10)
    A = [rng.randint(-128, 127) for _ in range(M * K)]
    Bm = [rng.randint(-128, 127) for _ in range(N * K)]   # B 以 [N][K] 存（K 連續）
    real = 40.0 / (math.sqrt(K) * 74.0 * 74.0)
    qm, sh = quantize_multiplier(real)
    golden = []
    for m in range(M):
        for n in range(N):
            acc = sum((A[m * K + k] - azp) * (Bm[n * K + k] - bzp) for k in range(K))
            golden.append(requantize(acc, qm, sh, ozp))
    sat = sum(1 for v in golden if v in (-128, 127))
    info = f"MACs={M*K*N}，飽和輸出 {100.0*sat/len(golden):.1f}%"
    body = c_defines({"T_M": M, "T_K": K, "T_N": N, "T_AZP": azp, "T_BZP": bzp,
                      "T_OZP": ozp, "T_QM": qm, "T_SHIFT": sh})
    body += c_array("int8_t", "t_a", A) + c_array("int8_t", "t_b", Bm)
    body += c_array("int8_t", "t_golden", golden)
    return body, info


GEN = {"gelu": gen_gelu, "add": gen_add, "softmax": gen_softmax,
       "layernorm": gen_layernorm, "matmul": gen_matmul}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=GEN.keys())
    ap.add_argument("--n", type=int, default=3072)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--L", type=int, default=197)
    ap.add_argument("--C", type=int, default=192)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--N", type=int, default=197)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    body, info = GEN[a.op](a, rng)
    with open(a.out, "w") as f:
        f.write(HEADER.format(cmd="gen_ops_test.py " + " ".join(sys.argv[1:])) + body)
    print(f"[gen] {a.out}: {a.op}  {info}")


if __name__ == "__main__":
    main()
