"""qref.py — 與 C kernel 逐位元一致的整數參考實作（Python）。

所有 gen_*_test.py 共用。這裡的函式只用整數運算，語意與 src/kernels/*.h / *.c 完全相同；
另外提供把結果寫成 C 陣列的工具。
"""
import math

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


# ---------------- 通用 ----------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def trunc_div(a, b):
    """C 的整數除法（向 0 截斷）。"""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def round_div(a, b):
    """四捨五入除法（b > 0），與 C 版 round_div 相同：遠離 0 的 half。"""
    return (a + b // 2) // b if a >= 0 else -((-a + b // 2) // b)


def wrap32(x):
    return ((x + (1 << 31)) % (1 << 32)) - (1 << 31)


# ---------------- requant（TFLite 語意）----------------
def quantize_multiplier(real):
    """real ≈ qm * 2^(shift-31)，qm ∈ [2^30, 2^31)。"""
    if real == 0.0:
        return 0, 0
    q, shift = math.frexp(real)
    qm = round(q * (1 << 31))
    if qm == (1 << 31):
        qm //= 2
        shift += 1
    return qm, shift


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
    return rdbpot(srdhm(wrap32(x << ls), qm), rs)


def requantize(acc, qm, shift, out_zp, lo=-128, hi=127):
    return clamp(mbqm(acc, qm, shift) + out_zp, lo, hi)


# ---------------- 量化工具（離線，允許浮點）----------------
def choose_qparams(vmin, vmax):
    """把實數範圍 [vmin, vmax] 對應到 int8，回傳 (scale, zero_point)。"""
    vmin, vmax = min(vmin, 0.0), max(vmax, 0.0)
    scale = (vmax - vmin) / 255.0 or 1.0
    zp = clamp(round(-128 - vmin / scale), -128, 127)
    return scale, zp


def quantize(v, scale, zp):
    return clamp(round(v / scale) + zp, -128, 127)


def dequantize(q, scale, zp):
    return (q - zp) * scale


# ---------------- 輸出 C ----------------
def c_array(ctype, name, vals, per_line=16):
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("  " + ", ".join(str(v) for v in vals[i:i + per_line]) + ",")
    return f"static const {ctype} {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def c_defines(d):
    return "".join(f"#define {k} {v}\n" for k, v in d.items())


def err_report(ints, floats_q):
    """整數近似結果 vs. 浮點算完再量化的結果，回傳 (最大誤差 LSB, 完全相同比例)。"""
    diffs = [abs(a - b) for a, b in zip(ints, floats_q)]
    return max(diffs), sum(1 for d in diffs if d == 0) / len(diffs)
