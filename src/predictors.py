"""Tüm modelleri AYNI arayüzle yükler: predictor.predict(normalize_edilmis_metinler) → (niyetler, güvenler).

Kalibrasyon (calibrate.py), ölçüm (benchmark.py) ve uygulama (app.py) modelleri buradan yükler.
Böylece üçü de aynı tahmin kodunu kullanır.

Her model klasöründe:
    thresholds.json   → calibrate.py'nin val setinde seçtiği iki eşik {"clinc": .., "custom": ..}
"""
import json
import os
from pathlib import Path

import joblib

from src.evaluate import ROOT

# SWE015_MODELS_DIR: başka bir model klasörünü (ör. ek eğitim verisi öncesi modeller) aynı kodla ölçmek için
MODELS = Path(os.environ.get("SWE015_MODELS_DIR", ROOT / "models"))
MODEL_DIRS = {
    "baseline": MODELS / "baseline",   # Model 1 · tf-idf + LR
    "ffn": MODELS / "ffn",             # Model 2 · fastText + FFN (SGD)
    "teacher": MODELS / "teacher",     # Model 3 · mBERT (öğretmen)
    "student": MODELS / "student",     # Model 4 · DistilmBERT (distillation)
    "student_int8": MODELS / "student_int8",  # Model 5 · DistilmBERT INT8 (quantized Transformer, uygulama)
    "teacher_int8": MODELS / "teacher_int8",  # Ek karşılaştırma · mBERT INT8
}


class BaselinePredictor:
    def __init__(self, path: Path):
        from src.train_baseline import predict
        self._predict = predict
        self.model = joblib.load(path / "model.joblib")
        self.single_threshold = json.loads((path / "config.json").read_text())["threshold"]

    def predict(self, texts_norm):
        return self._predict(self.model, list(texts_norm))


def load_predictor(name: str, device: str = "cpu"):
    path = MODEL_DIRS[name]
    if name == "baseline":
        pred = BaselinePredictor(path)
    elif name == "ffn":
        from src.train_ffn import FFNPredictor
        pred = FFNPredictor(path)
        pred.single_threshold = pred.threshold
    elif name.endswith("_int8"):
        from src.quantize import ONNXPredictor
        pred = ONNXPredictor(path)  # INT8 ONNX Runtime, CPU
        pred.single_threshold = pred.threshold
    else:
        from src.hf_model import HFPredictor
        pred = HFPredictor(path, device=device)
        pred.single_threshold = pred.threshold
    th_file = path / "thresholds.json"
    pred.thresholds = json.loads(th_file.read_text())["thresholds"] if th_file.exists() else None
    pred.name, pred.path = name, path
    return pred
