"""deit_data.py — DeiT-Tiny 的圖片前處理與 Imagenette 資料讀取（共用模組）。

前處理與 timm deit_tiny_patch16_224 的設定相同：
  短邊縮放到 256（= 224 / crop_pct 0.875，bicubic）→ 中央裁切 224 × 224
  → 除以 255 → 減 ImageNet mean、除以 std → NCHW float32
"""
import os

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG = 224
RESIZE = 256

# Imagenette 的 10 類（label 0~9）在 ImageNet-1k 1000 類中的編號
IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
IMAGENETTE_NAMES = ["tench", "English springer", "cassette player", "chain saw", "church",
                    "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "imagenette")


def preprocess(path):
    """回傳 (1, 3, 224, 224) float32。"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = RESIZE / min(w, h)
    img = img.resize((max(RESIZE, round(w * s)), max(RESIZE, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - IMG) // 2, (h - IMG) // 2
    img = img.crop((left, top, left + IMG, top + IMG))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def list_split(split, limit=None):
    """回傳 [(圖片路徑, ImageNet 類別編號), ...]，各類別輪流排列。"""
    root = os.path.join(DATA_DIR, split)
    per_class = []
    for label in range(10):
        d = os.path.join(root, str(label))
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        per_class.append([(os.path.join(d, f), IMAGENETTE_TO_IMAGENET[label]) for f in files])
    items = [it for group in zip(*per_class) for it in group]   # 交錯排列，前 N 張也會涵蓋所有類別
    return items[:limit] if limit else items
