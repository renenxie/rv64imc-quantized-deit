#!/usr/bin/env python3
"""計算模型每個運算的 MAC（乘加）次數，用來算「每條指令做幾個 MAC」。

    python3 scripts/count_macs.py models/deit_tiny_int8.tflite -o results/model_macs.txt

只算 FULLY_CONNECTED、BATCH_MATMUL、CONV_2D、DEPTHWISE_CONV_2D（真正做乘加的運算）。
輸出：每類運算的個數與 MAC 數、總 MAC 數；另附每個運算的 MAC（op 編號與 profile 的 op 編號相同），
      以及每個運算的輸出元素個數（ELEM 行，用來算「每個輸出元素花幾條指令」）。
需要：ai-edge-litert（只用它的 flatbuffer schema 讀模型）
"""
import argparse
import collections

import numpy as np
from ai_edge_litert import schema_py_generated as fb

NAMES = {v: k for k, v in vars(fb.BuiltinOperator).items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    m = fb.ModelT.InitFromPackedBuf(open(a.model, "rb").read(), 0)
    sg = m.subgraphs[0]
    shape = lambda t: [int(d) for d in sg.tensors[t].shape]

    per_op, per_type, elems = [], collections.OrderedDict(), []
    for i, op in enumerate(sg.operators):
        c = m.operatorCodes[op.opcodeIndex]
        name = NAMES[max(c.builtinCode, c.deprecatedBuiltinCode)]
        out = shape(op.outputs[0])
        elems.append((i, name, int(np.prod(out))))
        macs = 0
        if name == "FULLY_CONNECTED":
            w = shape(op.inputs[1])                      # [out_ch, in_dim]
            macs = int(np.prod(out[:-1])) * w[0] * w[1]
        elif name == "BATCH_MATMUL":
            lhs = shape(op.inputs[0])
            adj_x = bool(op.builtinOptions.adjX) if op.builtinOptions else False
            k = lhs[-2] if adj_x else lhs[-1]
            macs = int(np.prod(out)) * k
        elif name == "CONV_2D":
            w = shape(op.inputs[1])                      # [out_ch, kh, kw, in_ch]
            macs = int(np.prod(out)) * w[1] * w[2] * w[3]
        elif name == "DEPTHWISE_CONV_2D":
            w = shape(op.inputs[1])                      # [1, kh, kw, ch]
            macs = int(np.prod(out)) * w[1] * w[2]
        if macs:
            per_op.append((i, name, macs))
            n, s = per_type.get(name, (0, 0))
            per_type[name] = (n + 1, s + macs)

    total = sum(s for _, s in per_type.values())
    lines = ["# 模型 MAC 數", f"model={a.model}"]
    for name, (n, s) in per_type.items():
        lines.append(f"TYPE,{name},{n},{s}")
    lines.append(f"TOTAL_MACS,{total}")
    lines += [f"OP,{i},{name},{s}" for i, name, s in per_op]
    lines += [f"ELEM,{i},{name},{n}" for i, name, n in elems]
    text = "\n".join(lines) + "\n"
    print("\n".join(lines[:len(per_type) + 3]))
    if a.out:
        open(a.out, "w").write(text)
        print(f"已輸出 {a.out}")


if __name__ == "__main__":
    main()
