#!/usr/bin/env python3
"""從 Imagenette 驗證集（parquet）挑出校正用與評估用的圖片，存成 JPEG。

    python3 scripts/prepare_imagenette.py ~/imagenette_val.parquet --calib 20 --eval 100

資料來源：https://huggingface.co/datasets/Multimodal-Fatima/Imagenette_validation
  （Imagenette = ImageNet 中 10 個類別的子集，共 3925 張驗證圖片）
輸出：data/imagenette/calib/<label>/*.jpg、data/imagenette/eval/<label>/*.jpg
  每類取前 calib 張做校正、接下來 eval 張做評估，兩者不重疊。
"""
import argparse
import os

import pyarrow.parquet as pq

from deit_data import DATA_DIR, IMAGENETTE_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--calib", type=int, default=20, help="每類幾張校正圖片")
    ap.add_argument("--eval", type=int, default=100, help="每類幾張評估圖片")
    a = ap.parse_args()

    t = pq.read_table(a.parquet, columns=["image", "label", "id"]).to_pydict()
    rows = sorted(zip(t["id"], t["label"], t["image"]), key=lambda r: r[0])
    count = {s: [0] * 10 for s in ("calib", "eval")}
    for _id, label, img in rows:
        if count["calib"][label] < a.calib:
            split = "calib"
        elif count["eval"][label] < a.eval:
            split = "eval"
        else:
            continue
        d = os.path.join(DATA_DIR, split, str(label))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{_id:05d}.jpg"), "wb") as f:
            f.write(img["bytes"])
        count[split][label] += 1

    for s in ("calib", "eval"):
        print(f"{s}: {sum(count[s])} 張  " +
              ", ".join(f"{IMAGENETTE_NAMES[i]}={c}" for i, c in enumerate(count[s])))


if __name__ == "__main__":
    main()
