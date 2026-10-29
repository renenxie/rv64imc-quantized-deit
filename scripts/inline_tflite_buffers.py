#!/usr/bin/env python3
"""把 .tflite 中「存在檔案尾端、以 offset 指向」的權重，搬回 flatbuffer 內部的標準位置。

    python3 scripts/inline_tflite_buffers.py models/deit_tiny_int8.tflite   # 直接改寫該檔

原因：AI Edge Quantizer 輸出的模型把權重放在檔案尾端（Buffer.offset / Buffer.size），
      電腦版 LiteRT 支援這種格式，但 LiteRT Micro 不支援——它會把這些權重當成空的，
      推論結果就會錯。搬回標準位置後，兩者讀到的權重完全相同。
權重內容、運算圖、量化參數都不變。
"""
import argparse

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as fb


def inline_buffers(raw: bytes) -> bytes:
    m = fb.ModelT.InitFromPackedBuf(raw, 0)
    moved = 0
    for b in m.buffers:
        if b.offset and b.offset > 1:
            b.data = np.frombuffer(raw[b.offset:b.offset + b.size], dtype=np.uint8)
            b.offset, b.size = 0, 0
            moved += 1
    builder = flatbuffers.Builder(len(raw) + 1024)
    builder.Finish(m.Pack(builder), file_identifier=b"TFL3")
    print(f"權重搬回 flatbuffer 內部：{moved} 個 buffer")
    return bytes(builder.Output())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-o", "--out", default=None, help="輸出檔（預設覆寫輸入檔）")
    a = ap.parse_args()
    raw = open(a.model, "rb").read()
    out = inline_buffers(raw)
    with open(a.out or a.model, "wb") as f:
        f.write(out)


if __name__ == "__main__":
    main()
