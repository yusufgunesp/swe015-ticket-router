"""Model 4 · D · Knowledge distillation: mBERT (öğretmen) → DistilmBERT (öğrenci).

Çalıştırma:  python -m src.distill_student      (önce src.train_teacher çalışmış olmalı)
Çıktılar:    models/student/ (ağırlıklar + tokenizer + router_config.json)
Sonra:       python -m src.calibrate student   (iki eşik + test değerlendirmesi)
Bu dosya test setini hiç okumaz.

Öğrenci: distilbert-base-multilingual-cased. mBERT'ten daha az Transformer katmanı var (sayılar
benchmark.py'de model config'inden okunur), mBERT ile aynı tokenizer'ı kullanıyor. Eğitimde iki hedefi birlikte öğrenir (bkz. hf_model.train_classifier):
  1. Gerçek etiket (cross-entropy)
  2. Öğretmenin yumuşatılmış olasılık dağılımı (KL divergence, sıcaklık T=2)
Öğretmen çıktıları (logits) train cümleleri için BİR KEZ hesaplanıp saklanır. Böylece iki model
aynı anda GPU belleğinde durmaz.
"""
import gc

import numpy as np
import torch

from src.evaluate import ROOT, load_split
from src.hf_model import HFPredictor, train_classifier

STUDENT_NAME = "distilbert-base-multilingual-cased"
TEACHER_DIR = ROOT / "models" / "teacher"
OUT = ROOT / "models" / "student"
T, ALPHA = 2.0, 0.5


def main() -> None:
    train, val = load_split("train"), load_split("val")

    # 1) Öğretmenin train cümlelerindeki logit'leri (sadece TRAIN, sızıntı yok)
    teacher = HFPredictor(TEACHER_DIR, device="cuda")
    t_logits = teacher.logits(train["text_norm"].tolist())
    t_labels = teacher.labels.tolist()
    agree = (np.array(t_labels, dtype=object)[t_logits.argmax(1)] == train["intent"].values).mean()
    print(f"Öğretmen logit'leri: {t_logits.shape}, train'de öğretmen doğruluğu {agree:.4f}")
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # 2) Öğrenciyi distillation ile eğit (etiket sırası öğretmeninkiyle aynı)
    train_classifier(STUDENT_NAME, OUT, train, val, epochs=5, lr=5e-5, batch_size=32,
                     teacher_logits=t_logits, T=T, alpha=ALPHA, teacher_labels=t_labels)
    print("Kaydedildi:", OUT)


if __name__ == "__main__":
    main()
