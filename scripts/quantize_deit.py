#!/usr/bin/env python3
"""用 Google AI Edge Quantizer 把 FP32 的 DeiT-Tiny .tflite 量化成「完全整數」的 W8A8 模型。

    python3 scripts/quantize_deit.py models/deit_tiny_fp32.tflite models/deit_tiny_int8.tflite

需要：pip install ai-edge-quantizer pillow pyarrow
校正資料：data/imagenette/calib（由 prepare_imagenette.py 產生）
recipe：static_wi8_ai8 = 權重 INT8、activation INT8，所有支援的運算都量化

量化前先對 FP32 模型做兩個「不改變計算結果」的修正：
  1. 重複的 tensor 名稱加上編號（量化工具要求名稱唯一）
  2. SQUEEZE → 等效的 RESHAPE
     量化工具不支援 INT8 的 SQUEEZE，會把它留成浮點並前後插入 DEQUANTIZE / QUANTIZE；
     SQUEEZE 只是拿掉長度 1 的維度，與 RESHAPE 成「同一個輸出形狀」完全等價，
     而 RESHAPE 可以量化成 INT8。
量化後：
  3. 把權重從檔案尾端搬回 flatbuffer 內部（LiteRT Micro 只支援這種格式）
  4. 檢查模型中不得有任何浮點（不得有 FP 運算）
"""
import argparse
import collections
import sys
import time

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as fb
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from ai_edge_quantizer import quantizer, recipe

from deit_data import list_split, preprocess
from inline_tflite_buffers import inline_buffers


def _opcode(op_code):
    return max(op_code.builtinCode, op_code.deprecatedBuiltinCode)


def _get_or_add_opcode(m, builtin):
    for i, c in enumerate(m.operatorCodes):
        if _opcode(c) == builtin:
            return i
    c = fb.OperatorCodeT()
    c.builtinCode = builtin
    c.deprecatedBuiltinCode = min(builtin, 127)
    c.version = 1
    m.operatorCodes.append(c)
    return len(m.operatorCodes) - 1


def uniquify_tensor_names(m):
    renamed = 0
    for sg in m.subgraphs:
        seen = collections.Counter()
        for t in sg.tensors:
            name = t.name.decode()
            seen[name] += 1
            if seen[name] > 1:
                t.name = f"{name}_{seen[name] - 1}".encode()
                renamed += 1
    return renamed


def squeeze_to_reshape(m):
    reshape_idx = _get_or_add_opcode(m, fb.BuiltinOperator.RESHAPE)
    replaced = 0
    for sg in m.subgraphs:
        for op in sg.operators:
            if _opcode(m.operatorCodes[op.opcodeIndex]) != fb.BuiltinOperator.SQUEEZE:
                continue
            out_shape = [int(d) for d in sg.tensors[op.outputs[0]].shape]
            # 新增一個 INT32 常數 tensor 存放目標形狀（RESHAPE 的第二個輸入）
            buf = fb.BufferT()
            buf.data = np.array(out_shape, dtype="<i4").view(np.uint8)
            m.buffers.append(buf)
            t = fb.TensorT()
            t.shape = [len(out_shape)]
            t.type = fb.TensorType.INT32
            t.buffer = len(m.buffers) - 1
            t.name = f"{sg.tensors[op.outputs[0]].name.decode()}_shape".encode()
            sg.tensors.append(t)
            # 把運算換成 RESHAPE
            op.opcodeIndex = reshape_idx
            op.inputs = [op.inputs[0], len(sg.tensors) - 1]
            opts = fb.ReshapeOptionsT()
            opts.newShape = out_shape
            op.builtinOptionsType = fb.BuiltinOptions.ReshapeOptions
            op.builtinOptions = opts
            replaced += 1
    return replaced


def prepare_fp32(path):
    m = fb.ModelT.InitFromPackedBuf(open(path, "rb").read(), 0)
    n1 = uniquify_tensor_names(m)
    n2 = squeeze_to_reshape(m)
    b = flatbuffers.Builder(1024)
    b.Finish(m.Pack(b), file_identifier=b"TFL3")
    print(f"前處理：重新命名重複 tensor {n1} 個；SQUEEZE → RESHAPE {n2} 個")
    return bytearray(b.Output())


def _interp(model):
    kw = {"model_content": bytes(model)} if isinstance(model, (bytes, bytearray)) else {"model_path": model}
    it = Interpreter(**kw, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    return it


def check_same_fp32(orig_path, new_model, n=3):
    """確認修改後的 FP32 模型與原本輸出完全相同。"""
    a, b = _interp(orig_path), _interp(new_model)
    for p, _ in list_split("calib", n):
        x = preprocess(p)
        outs = []
        for it in (a, b):
            it.set_tensor(it.get_input_details()[0]["index"], x)
            it.invoke()
            outs.append(it.get_tensor(it.get_output_details()[0]["index"]).copy())
        if not np.array_equal(outs[0], outs[1]):
            sys.exit("錯誤：修改後的 FP32 模型輸出與原本不同")
    print(f"確認：修改後的 FP32 模型與原本輸出完全相同（{n} 張圖片）")


def check_no_float(path):
    """量化後的模型中，任何運算的 activation 輸入/輸出都不得是浮點。"""
    it = _interp(path)
    td = {t["index"]: t for t in it.get_tensor_details()}
    bad = collections.Counter()
    for o in it._get_ops_details():
        for i in list(o["inputs"]) + list(o["outputs"]):
            if i >= 0 and np.issubdtype(td[i]["dtype"], np.floating):
                bad[o["op_name"]] += 1
                break
    if bad:
        sys.exit(f"錯誤：仍有浮點運算：{dict(bad)}")
    print(f"確認：{len(it._get_ops_details())} 個運算全部沒有浮點")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fp32_model")
    ap.add_argument("out_model")
    ap.add_argument("--calib", type=int, default=200, help="校正圖片張數")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    fp32 = prepare_fp32(a.fp32_model)
    check_same_fp32(a.fp32_model, fp32)

    qt = quantizer.Quantizer(fp32)
    qt.load_quantization_recipe(recipe.static_wi8_ai8())

    items = list_split("calib", a.calib)
    print(f"校正：{len(items)} 張圖片")
    samples = [{"x": preprocess(p)} for p, _ in items]

    t0 = time.time()
    calib = qt.calibrate({"main": samples}, num_threads=a.threads)
    print(f"校正完成（{time.time()-t0:.0f}s），開始量化")
    result = qt.quantize(calib, enable_progress_bar=False)
    result.export_model(a.out_model, overwrite=True)
    raw = open(a.out_model, "rb").read()
    with open(a.out_model, "wb") as f:
        f.write(inline_buffers(raw))
    print(f"已輸出 {a.out_model}")
    check_no_float(a.out_model)


if __name__ == "__main__":
    main()
