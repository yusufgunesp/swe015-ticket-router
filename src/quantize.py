"""Model 5 · D · Quantization: Transformer'ları ONNX'e çevirip ONNX Runtime ile dinamik INT8'e quantize etme.

Çalıştırma:  python -m src.quantize            (öğrenci → student_int8, öğretmen → teacher_int8)
Sonra:       python -m src.calibrate student_int8 teacher_int8

Yöntem: post-training dynamic quantization (yeniden eğitim yok), ONNX Runtime (CPU).
  1. PyTorch modeli ONNX biçimine dışa aktarılır (FP32).
  2. onnxruntime.quantization.quantize_dynamic:
     - MatMul ağırlıkları (Linear katmanlar) → 8 bit, tensör başına ölçek. Aktivasyonlar çalışma anında
       her girdi için dinamik olarak 8 bite çevrilir. Çarpım tam sayı aritmetiğiyle yapılır.
     - Gather (kelime/konum gömme tabloları) → 8 bit. Modelin en büyük parçası bu tablolar.
     - MatMulConstBOnly: Sadece sabit ağırlıklı MatMul'lar quantize edilir. Attention içindeki
       aktivasyon × aktivasyon çarpımları (Q·Kᵀ, skor·V) FP32 kalır.
     - LayerNorm, softmax, GELU FP32 kalır.
  3. Ağırlık tipi QUInt8 (UInt8): Bu işlemcide QInt8 ağırlık val doğruluğunu belirgin şekilde
     düşürdü, QUInt8 ise FP32'ye çok yakın kaldı. Sayılar: results/quantization_experiments.csv
     (üreten script: src/quant_experiments.py). Olası neden: AVX2 işlemcide (VNNI yok) işaretli
     INT8 çarpımında taşma. Bu neden test edilmedi.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.evaluate import ROOT
from src.hf_model import MAX_LEN

warnings.filterwarnings("ignore", category=DeprecationWarning)

PAIRS = {"student_int8": "student", "teacher_int8": "teacher"}
ONNX_FILE = "model.onnx"
QUANT_KW = dict(weight_type=QuantType.QUInt8, per_channel=False, op_types_to_quantize=["MatMul", "Gather"],
                extra_options={"MatMulConstBOnly": True})


def export_onnx(src_dir: Path, out_file: Path) -> None:
    """PyTorch (FP32) → ONNX (FP32). Dinamik eksenler: batch ve cümle uzunluğu değişebilir."""
    tok = AutoTokenizer.from_pretrained(src_dir)
    model = AutoModelForSequenceClassification.from_pretrained(src_dir).eval()
    enc = tok(["örnek bir cümle"], return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), str(out_file), dynamo=False,
                          input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                        "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
                          opset_version=17)


def quantize(dst: str) -> None:
    src_dir, dst_dir = ROOT / "models" / PAIRS[dst], ROOT / "models" / dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # ONNX'in shape-inference adımı Türkçe karakterli yollarda ("Masaüstü") ara dosyayı açamıyor.
    # Bu yüzden dönüştürme ASCII yollu geçici bir klasörde yapılıp sonuç taşınır.
    with tempfile.TemporaryDirectory() as tmp:
        fp32_file, int8_file = Path(tmp) / "model_fp32.onnx", Path(tmp) / ONNX_FILE
        export_onnx(src_dir, fp32_file)
        quantize_dynamic(str(fp32_file), str(int8_file), **QUANT_KW)
        shutil.move(str(int8_file), dst_dir / ONNX_FILE)  # sadece INT8 model saklanır
    # Tokenizer, config ve eğitimdeki tek eşik bilgisi kopyalanır. thresholds.json kopyalanmaz:
    # INT8 modelin eşikleri calibrate.py ile val setinde YENİDEN seçilir.
    for f in src_dir.iterdir():
        if f.name not in {"model.safetensors", "pytorch_model.bin", "thresholds.json"}:
            shutil.copy(f, dst_dir / f.name)
    (dst_dir / "quantization.json").write_text(json.dumps({
        "source": PAIRS[dst], "method": "ONNX Runtime post-training dynamic quantization",
        "weights": "UInt8, per-tensor (MatMul with constant B + Gather/embeddings)",
        "activations": "UInt8, computed dynamically per input",
        "fp32_ops": "LayerNorm, Softmax, GELU, attention activation x activation MatMuls",
        "onnxruntime": ort.__version__, "torch": torch.__version__}, indent=2))
    print(f"{PAIRS[dst]} → {dst}: {(dst_dir / ONNX_FILE).stat().st_size / 2**20:.1f} MB | "
          f"{time.time() - t0:.0f} sn", flush=True)


class ONNXPredictor:
    """INT8 ONNX modeli CPU'da çalıştırır. predict() arayüzü HFPredictor ile aynı (aynı tokenizer, MAX_LEN)."""

    def __init__(self, path: Path, threads: int | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        cfg = AutoConfig.from_pretrained(path)
        self.labels = np.array([cfg.id2label[i] for i in range(cfg.num_labels)], dtype=object)
        opts = ort.SessionOptions()
        # Thread sayısı PyTorch ile aynı (fiziksel çekirdek sayısı): adil gecikme karşılaştırması
        opts.intra_op_num_threads = threads or torch.get_num_threads()
        # Model bayt olarak verilir: Türkçe karakterli yollarda dosya açma sorunu olmasın
        self.session = ort.InferenceSession((path / ONNX_FILE).read_bytes(), opts,
                                            providers=["CPUExecutionProvider"])
        rc = path / "router_config.json"
        self.threshold = json.loads(rc.read_text())["threshold"] if rc.exists() else None

    def logits(self, texts) -> np.ndarray:
        """HER CÜMLE TEK BAŞINA çalıştırılır (batch=1).
        Dinamik quantization'da aktivasyon ölçeği girdi tensörünün tamamından hesaplanır. Toplu (batch)
        çalıştırınca bir cümlenin sonucu yanındaki cümlelere ve dolguya bağlı oluyordu (doğruluk denetimi:
        testte batch=128 ile batch=1 arasında 155/5560 tahmin farklıydı). Uygulama mesajları tek tek
        işlediği için değerlendirme de tek tek yapılır."""
        out = []
        for t in texts:
            enc = self.tokenizer([t], truncation=True, max_length=MAX_LEN, return_tensors="np")
            out.append(self.session.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                               "attention_mask": enc["attention_mask"].astype(np.int64)})[0])
        return np.concatenate(out)

    def predict(self, texts, batch_size: int = 1):  # batch_size sadece arayüz uyumu için, her zaman 1
        z = self.logits(texts)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        proba = z / z.sum(axis=1, keepdims=True)  # softmax
        idx = proba.argmax(axis=1)
        return self.labels[idx], proba[np.arange(len(idx)), idx]


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for dst in sys.argv[1:] or list(PAIRS):
        quantize(dst)


if __name__ == "__main__":
    main()
