#!/usr/bin/env python3
"""產生給 RISC-V 端 LiteRT Micro 用的輸入圖片與參考答案。

    python3 scripts/make_runtime_io.py models/deit_tiny_int8.tflite runtime/build/io [--index 0]

輸出：
  input.bin   一張評估圖片，前處理後量化成 INT8（形狀 1×3×224×224，與模型輸入相同）
  golden.bin  Python LiteRT 用「參考 kernel」（BUILTIN_REF）執行的 INT8 輸出（1000 類）
  info.txt    圖片路徑、正確類別、參考答案的 top-1

參考 kernel 是 TFLite 最直接、未最佳化的整數實作，與 LiteRT Micro 的 kernel 同源，
因此 RISC-V 上的輸出應與 golden.bin 逐位元相同。
"""
import argparse
import os
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deit_data import list_split, preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--index", type=int, default=0, help="使用評估集的第幾張圖片")
    a = ap.parse_args()

    path, label = list_split("eval")[a.index]
    it = Interpreter(model_path=a.model, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    din, dout = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = din["quantization"]
    x = np.clip(np.round(preprocess(path) / scale) + zp, -128, 127).astype(np.int8)
    it.set_tensor(din["index"], x)
    it.invoke()
    y = it.get_tensor(dout["index"]).astype(np.int8)

    os.makedirs(a.out_dir, exist_ok=True)
    x.tofile(os.path.join(a.out_dir, "input.bin"))
    y.tofile(os.path.join(a.out_dir, "golden.bin"))
    top1 = int(np.argmax(y))
    with open(os.path.join(a.out_dir, "info.txt"), "w") as f:
        f.write(f"image={path}\nlabel={label}\ngolden_top1={top1}\n")
    print(f"圖片 {os.path.basename(path)}：正確類別 {label}，參考答案 top-1 = {top1}，"
          f"input {x.nbytes} bytes，golden {y.nbytes} bytes")


if __name__ == "__main__":
    main()
