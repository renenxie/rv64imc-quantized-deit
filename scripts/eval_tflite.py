#!/usr/bin/env python3
"""用 LiteRT（Python）量測 .tflite 模型在 Imagenette 評估集上的準確率。

    python3 scripts/eval_tflite.py models/deit_tiny_fp32.tflite
    python3 scripts/eval_tflite.py models/deit_tiny_int8.tflite --limit 200

準確率以 ImageNet 1000 類計算（模型要從 1000 類中選出正確的那一類）。
若模型輸入/輸出是整數，會自動用模型內的 scale / zero point 做量化與反量化。
"""
import argparse
import os
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from deit_data import list_split, preprocess


def to_input(x, det):
    if det["dtype"] == np.float32:
        return x
    scale, zp = det["quantization"]
    info = np.iinfo(det["dtype"])
    return np.clip(np.round(x / scale) + zp, info.min, info.max).astype(det["dtype"])


def from_output(y, det):
    if det["dtype"] == np.float32:
        return y
    scale, zp = det["quantization"]
    return (y.astype(np.float32) - zp) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("-o", "--out", default=None, help="結果另存的文字檔")
    a = ap.parse_args()

    it = Interpreter(a.model, num_threads=a.threads)
    it.allocate_tensors()
    din, dout = it.get_input_details()[0], it.get_output_details()[0]
    items = list_split(a.split, a.limit)
    if not items:
        raise SystemExit("找不到圖片，請先執行 prepare_imagenette.py")

    top1 = top5 = 0
    t0 = time.time()
    for i, (path, label) in enumerate(items, 1):
        it.set_tensor(din["index"], to_input(preprocess(path), din))
        it.invoke()
        logits = from_output(it.get_tensor(dout["index"]), dout)[0]
        rank = np.argsort(-logits)
        top1 += int(rank[0] == label)
        top5 += int(label in rank[:5])
        if i % 100 == 0 or i == len(items):
            print(f"  {i}/{len(items)}  top-1 {100*top1/i:.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    n = len(items)
    line = (f"model={os.path.basename(a.model)} input={np.dtype(din['dtype']).name} "
            f"images={n} top1={100*top1/n:.2f}% top5={100*top5/n:.2f}%")
    print(line)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
