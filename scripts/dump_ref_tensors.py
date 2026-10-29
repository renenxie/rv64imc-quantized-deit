#!/usr/bin/env python3
"""除錯工具：用 Python LiteRT 參考 kernel 跑一次，列出每個運算輸出 tensor 的檢查碼，
與 RISC-V 上 LiteRT Micro 的結果（make -C runtime dump）逐個對照，找出第一個不一致的運算。

    python3 scripts/dump_ref_tensors.py models/deit_tiny_int8.tflite runtime/build/io/input.bin > ref.txt
    python3 scripts/dump_ref_tensors.py ... --compare runtime/build/dump_rv.txt   # 直接比對
    python3 scripts/dump_ref_tensors.py ... --save 37 out.npz                       # 存下第 37 個運算的輸入/輸出

輸出格式與 runtime/main.cc 的 DUMP 模式相同：T <op 編號> <tensor 編號> <bytes> <FNV-1a 32 位元>
"""
import argparse

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def fnv1a(b: bytes) -> int:
    h = 2166136261
    for x in b:
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("input_bin")
    ap.add_argument("--compare", help="RISC-V 端的 dump 輸出")
    ap.add_argument("--save", nargs=2, metavar=("OP", "NPZ"), help="存下某個運算的輸入/輸出")
    a = ap.parse_args()

    it = Interpreter(model_path=a.model, experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                     experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    din = it.get_input_details()[0]
    x = np.fromfile(a.input_bin, dtype=np.int8).reshape(din["shape"])
    it.set_tensor(din["index"], x)
    it.invoke()
    ops = it._get_ops_details()

    lines = []
    for i, o in enumerate(ops):
        t = int(o["outputs"][0])
        b = it.get_tensor(t).tobytes()
        lines.append(f"T {i} {t} {len(b)} {fnv1a(b):08x}")

    if a.save:
        k = int(a.save[0])
        o = ops[k]
        td = {d["index"]: d for d in it.get_tensor_details()}
        out = {"op_name": o["op_name"]}
        for j, t in enumerate(list(o["inputs"]) + list(o["outputs"])):
            if t < 0:
                continue
            tag = f"in{j}" if j < len(o["inputs"]) else f"out{j - len(o['inputs'])}"
            out[tag] = it.get_tensor(t)
            out[tag + "_q"] = np.array(td[t]["quantization"], dtype=np.float64)
            out[tag + "_name"] = td[t]["name"]
        np.savez(a.save[1], **out)
        print(f"已存下 op {k}（{o['op_name']}）到 {a.save[1]}")

    if a.compare:
        rv = {}
        for ln in open(a.compare, encoding="utf-8"):
            if ln.startswith("T "):
                _, i, t, n, h = ln.split()
                rv[int(i)] = (int(t), int(n), h)
        bad = 0
        for i, ln in enumerate(lines):
            _, _, t, n, h = ln.split()
            r = rv.get(i)
            if r is None or r[2] != h:
                print(f"不一致：op {i} {ops[i]['op_name']} tensor {t}（Python {h} / RISC-V {r[2] if r else '無'}）")
                bad += 1
                if bad >= 15:
                    break
        print(f"共 {len(lines)} 個運算，前 15 個內不一致：{bad}")
    elif not a.save:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
