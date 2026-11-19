"""D · Quantization yöntem karşılaştırması (öğrenci modeli, SADECE val seti). Hangi INT8 yöntemi neden seçildi?

Çalıştırma:  python -m src.quant_experiments
Çıktı:       results/quantization_experiments.csv

Varyantlar: PyTorch FP32, PyTorch dinamik INT8 (tensör / kanal başına), ONNX Runtime FP32,
ONNX Runtime dinamik INT8 (QInt8 / QUInt8 ağırlık). Ölçülenler: val doğruluğu (eşiksiz, en olası niyet,
her cümle tek başına), serileştirilmiş boyut (MB, sadece ağırlıklar: tokenizer YOK), tek mesaj gecikmesi
(batch=1, normalize edilmiş val cümleleri, normalizasyon süresi YOK, 10 ısınma + 200 çalıştırma).
Bu yüzden MB ve ms değerleri results.csv ile aynı tabloda karıştırılmamalı.
Test seti burada KULLANILMAZ.
"""
import copy
import io
import tempfile
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from transformers import AutoModelForSequenceClassification

from src.evaluate import RESULTS, ROOT, load_split
from src.hf_model import HFPredictor
from src.quantize import ONNX_FILE, QUANT_KW, ONNXPredictor, export_onnx

warnings.filterwarnings("ignore")
SRC = ROOT / "models" / "student"


def measure(pred, texts, labels_true) -> dict:
    # batch=1: dinamik quantization sonucu batch içeriğine bağlı. Uygulamadaki gibi tek tek tahmin
    lab, _ = pred.predict(texts, batch_size=1)
    for t in texts[:10]:
        pred.predict([t])
    t0 = time.perf_counter()
    for t in texts[:200]:
        pred.predict([t])
    return {"val_acc_no_threshold": float((lab == labels_true).mean()),
            "latency_ms": (time.perf_counter() - t0) / 200 * 1000}


def main() -> None:
    from torch.ao.quantization import (default_dynamic_qconfig, float_qparams_weight_only_qconfig,
                                       per_channel_dynamic_qconfig, quantize_dynamic as torch_qd)
    v = load_split("val")
    texts, y = v["text_norm"].tolist(), v["intent"].values
    base = AutoModelForSequenceClassification.from_pretrained(SRC).eval()
    rows = []

    torch_variants = {
        "PyTorch FP32": None,
        "PyTorch dynamic INT8, per-tensor Linear + 8-bit embeddings":
            {nn.Linear: default_dynamic_qconfig, nn.Embedding: float_qparams_weight_only_qconfig},
        "PyTorch dynamic INT8, per-channel Linear + 8-bit embeddings":
            {nn.Linear: per_channel_dynamic_qconfig, nn.Embedding: float_qparams_weight_only_qconfig},
    }
    for name, spec in torch_variants.items():
        m = copy.deepcopy(base)
        if spec:
            m = torch_qd(m, spec, dtype=torch.qint8)
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        r = measure(HFPredictor(SRC, device="cpu", model=m), texts, y)
        rows.append({"variant": name, "size_mb": buf.tell() / 2**20, **r})
        print(rows[-1], flush=True)

    with tempfile.TemporaryDirectory() as tmp:  # ASCII yol (bkz. quantize.py)
        tmp = Path(tmp)
        for f in SRC.iterdir():  # tokenizer + config
            if f.suffix == ".json":
                (tmp / f.name).write_bytes(f.read_bytes())
        fp32 = tmp / "fp32.onnx"
        export_onnx(SRC, fp32)
        onnx_variants = {
            "ONNX Runtime FP32": None,
            "ONNX Runtime dynamic INT8, QInt8 weights": {**QUANT_KW, "weight_type": QuantType.QInt8},
            "ONNX Runtime dynamic INT8, QUInt8 weights (chosen)": QUANT_KW,
        }
        for name, kw in onnx_variants.items():
            target = tmp / ONNX_FILE
            if kw is None:
                target.write_bytes(fp32.read_bytes())
            else:
                quantize_dynamic(str(fp32), str(target), **kw)
            r = measure(ONNXPredictor(tmp), texts, y)
            rows.append({"variant": name, "size_mb": target.stat().st_size / 2**20, **r})
            print(rows[-1], flush=True)

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "quantization_experiments.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
