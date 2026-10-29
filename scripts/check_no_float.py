#!/usr/bin/env python3
"""檢查 .tflite 模型中「完全沒有浮點」，任何一項不通過就以非 0 結束。

    python3 scripts/check_no_float.py models/deit_tiny_int8.tflite

檢查項目：
  1. 模型的輸入、輸出都是整數型別
  2. 所有常數 tensor（權重、bias、查表、形狀參數…）都是整數型別
  3. 每個運算的所有輸入、輸出 tensor 都是整數型別
  4. 沒有浮點與整數之間的轉換運算（DEQUANTIZE，或輸入為浮點的 QUANTIZE）
說明：tensor 的 scale 本身是浮點數，但它只是記錄在模型檔裡的「量化參數」，
      推論時由事前換算好的整數 quant_mult / shift 取代，不會產生浮點運算，因此不列入檢查。
"""
import argparse
import collections
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def is_float(dtype):
    return np.issubdtype(np.dtype(dtype), np.floating)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    a = ap.parse_args()

    # 不使用 XNNPACK 等預設加速器，才能檢查模型原本的每一個運算
    it = Interpreter(model_path=a.model,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ops = it._get_ops_details()
    td = {t["index"]: t for t in it.get_tensor_details()}
    produced = {i for o in ops for i in o["outputs"]}
    model_in = {d["index"] for d in it.get_input_details()}

    fails = []

    # 1. 模型輸入、輸出
    for kind, dets in [("輸入", it.get_input_details()), ("輸出", it.get_output_details())]:
        for d in dets:
            if is_float(d["dtype"]):
                fails.append(f"模型{kind} {d['name']} 是 {np.dtype(d['dtype']).name}")

    # 2. 常數 tensor
    const_float = collections.Counter()
    for o in ops:
        for i in o["inputs"]:
            if i >= 0 and i not in produced and i not in model_in and is_float(td[i]["dtype"]):
                const_float[o["op_name"]] += 1
    for name, c in const_float.items():
        fails.append(f"{c} 個 {name} 使用浮點常數（權重或參數）")

    # 3. 每個運算的輸入/輸出
    op_float = collections.Counter()
    for o in ops:
        if any(i >= 0 and is_float(td[i]["dtype"]) for i in list(o["inputs"]) + list(o["outputs"])):
            op_float[o["op_name"]] += 1
    for name, c in op_float.items():
        fails.append(f"{c} 個 {name} 運算有浮點的輸入或輸出")

    # 4. 浮點 ↔ 整數轉換
    conv = sum(1 for o in ops if o["op_name"] == "DEQUANTIZE" or
               (o["op_name"] == "QUANTIZE" and is_float(td[o["inputs"][0]]["dtype"])))
    if conv:
        fails.append(f"{conv} 個浮點與整數之間的轉換運算（DEQUANTIZE / QUANTIZE）")

    dtypes = collections.Counter(np.dtype(td[i]["dtype"]).name
                                 for o in ops for i in list(o["inputs"]) + list(o["outputs"]) if i >= 0)
    print(f"模型：{a.model}")
    print(f"運算數：{len(ops)}；運算用到的 tensor 型別：{dict(dtypes)}")
    if fails:
        print("結果：FAIL，模型含有浮點")
        for f in fails:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("結果：PASS，模型完全沒有浮點（輸入、輸出、權重、所有運算皆為整數）")


if __name__ == "__main__":
    main()
