#!/usr/bin/env python3
"""檢查 .tflite 模型：用到哪些運算、每個運算的輸入/輸出型別、權重大小與是否量化。

    python3 scripts/inspect_tflite.py models/deit_tiny_int8.tflite          # 摘要
    python3 scripts/inspect_tflite.py models/deit_tiny_int8.tflite --ops    # 逐一列出每個運算

使用 LiteRT 官方 interpreter 判讀（pip install ai-edge-litert），運算名稱與型別最可靠。
"""
import argparse
import collections

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ops", action="store_true", help="逐一列出每個運算")
    a = ap.parse_args()

    # 不使用 XNNPACK 等預設加速器，才能看到模型原本的每一個運算
    it = Interpreter(a.model, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ops = it._get_ops_details()
    td = {t["index"]: t for t in it.get_tensor_details()}

    # 常數 tensor = 不是模型輸入、也不是任何運算輸出的 tensor（權重、bias、形狀參數等）
    produced = {i for o in ops for i in o["outputs"]}
    model_in = {d["index"] for d in it.get_input_details()}
    is_const = lambda i: i not in produced and i not in model_in

    def dt(i):
        return np.dtype(td[i]["dtype"]).name

    def quantized(i):
        return len(td[i]["quantization_parameters"]["scales"]) > 0

    print(f"運算數: {len(ops)}，tensor 數: {len(td)}")
    for kind, dets in [("輸入", it.get_input_details()), ("輸出", it.get_output_details())]:
        for d in dets:
            print(f"  {kind}: {d['name']}  {np.dtype(d['dtype']).name}  {[int(x) for x in d['shape']]}  "
                  f"量化={len(d['quantization_parameters']['scales']) > 0}")

    wbytes = collections.Counter()
    seen = set()
    for o in ops:
        for i in o["inputs"]:
            if i >= 0 and is_const(i) and i not in seen:
                seen.add(i)
                n = int(np.prod(td[i]["shape"])) if len(td[i]["shape"]) else 1
                wbytes[(dt(i), quantized(i))] += n * np.dtype(td[i]["dtype"]).itemsize
    print("\n常數 tensor（權重、bias 等）總大小：")
    for (d, q), b in sorted(wbytes.items(), key=lambda x: -x[1]):
        print(f"  {d:8s} 量化={str(q):5s}  {b/1e6:8.2f} MB")

    hist = collections.Counter()
    for o in ops:
        act = [dt(i) for i in o["inputs"] if i >= 0 and not is_const(i)]
        outs = [dt(i) for i in o["outputs"]]
        hist[(o["op_name"], "/".join(sorted(set(act))) or "-", "/".join(sorted(set(outs))))] += 1
    print("\n運算種類（activation 輸入型別 → 輸出型別）：")
    for (name, ti, to), c in sorted(hist.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {c:4d} × {name:18s} {ti:>8s} → {to}")

    if a.ops:
        print("\n逐一列出：")
        for o in ops:
            fmt = lambda ids: ", ".join(f"{dt(i)}{[int(x) for x in td[i]['shape']]}" for i in ids)
            act = [i for i in o["inputs"] if i >= 0 and not is_const(i)]
            cst = [i for i in o["inputs"] if i >= 0 and is_const(i)]
            print(f"  [{o['index']:3d}] {o['op_name']:16s} in: {fmt(act)}  const: {fmt(cst)}  out: {fmt(o['outputs'])}")


if __name__ == "__main__":
    main()
