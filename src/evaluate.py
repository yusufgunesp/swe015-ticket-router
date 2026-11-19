"""Tüm modeller için ORTAK değerlendirme kodu (R3: aynı test seti, aynı metrik, aynı karar kuralı).

Karar kuralı (uygulamada da aynısı kullanılır, bkz. router.py):
    tahmin = "oos" (→ human agent)   eğer model "oos" dediyse  VEYA  güven < eşik
    tahmin = modelin niyeti          aksi halde

İKİ AYRI EŞİK kullanılır: tahmin edilen niyet bizim beyaz eşya niyetlerimizden biriyse "custom"
eşiği, CLINC niyetlerinden biriyse "clinc" eşiği uygulanır.
Neden: Tek eşik 3.100 CLINC val cümlesine göre seçiliyordu (yalnızca 60'ı beyaz eşya). mBERT beyaz
eşya cümlelerinin %90'ını doğru tahmin ettiği halde güveni ~0.6 olduğu için, 0.81'lik ortak eşik bu
doğru tahminlerin çoğunu human agent'a gönderiyordu. (19 Kas'ta, mBERT'in test sonucu görüldükten
sonra fark edildi. Teşhis ve seçim val setinde yapıldı.)

Eşikler SADECE validation setinde seçilir (calibrate_thresholds). Test seti eşik seçiminde kullanılmaz.
Eğitim sırasında epoch/C seçimi için tek eşikli choose_threshold kullanılır (model seçim ölçütü).

Metrikler:
    in_scope_acc : gerçek etiketi "oos" OLMAYAN test örneklerinde doğru niyet oranı
                   (in-scope bir örnek yanlışlıkla human agent'a giderse YANLIŞ sayılır)
    oos_recall   : gerçek etiketi "oos" olan test örneklerinden kaçının "oos" tahmin edildiği
    custom_acc   : bizim beyaz eşya test cümlelerimizde (TR+EN) doğru niyet oranı
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.normalize import normalize_text

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
OOS = "oos"


def split_sha256(split: str) -> str:
    return hashlib.sha256((PROCESSED / f"{split}.csv").read_bytes()).hexdigest()


def check_split_integrity() -> None:
    """Bölmeler data.py'nin ürettiğinden sonra değiştiyse hata ver."""
    info = json.loads((PROCESSED / "SPLIT_INFO.json").read_text(encoding="utf-8"))
    for split, meta in info["splits"].items():
        if split_sha256(split) != meta["sha256"]:
            raise RuntimeError(f"{split}.csv değişmiş! SPLIT_INFO.json ile eşleşmiyor.")


@lru_cache(maxsize=3)
def load_split(split: str) -> pd.DataFrame:
    """Bölmeyi okur ve normalize edilmiş metni 'text_norm' sütununa ekler."""
    check_split_integrity()
    df = pd.read_csv(PROCESSED / f"{split}.csv")
    df["text_norm"] = df["text"].map(normalize_text)
    return df


CUSTOM_INTENTS = frozenset({"dishwasher_issue", "washing_machine_issue", "refrigerator_issue",
                            "warranty_inquiry", "technician_appointment"})


def decide(pred: np.ndarray, conf: np.ndarray, threshold) -> np.ndarray:
    """Karar kuralı: düşük güven ya da 'oos' → 'oos' (human agent).
    threshold: tek sayı  VEYA  {"clinc": t1, "custom": t2} (tahmin edilen niyete göre eşik)."""
    pred = np.asarray(pred, dtype=object).copy()
    conf = np.asarray(conf)
    if isinstance(threshold, dict):
        is_custom = np.fromiter((p in CUSTOM_INTENTS for p in pred), dtype=bool, count=len(pred))
        thr = np.where(is_custom, threshold["custom"], threshold["clinc"])
    else:
        thr = threshold
    pred[conf < thr] = OOS
    return pred


def compute_metrics(y_true, y_pred, source=None) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    ins = y_true != OOS
    out = {
        "in_scope_acc": float((y_pred[ins] == y_true[ins]).mean()),
        "oos_recall": float((y_pred[~ins] == OOS).mean()),
        "overall_acc": float((y_pred == y_true).mean()),
        "n_in_scope": int(ins.sum()),
        "n_oos": int((~ins).sum()),
    }
    if source is not None:
        cm = np.asarray(source) == "custom"
        correct = int((y_pred[cm] == y_true[cm]).sum())
        out["custom_acc"] = correct / int(cm.sum())
        out["n_custom"] = int(cm.sum())
        # Sadece 60 beyaz eşya test cümlesi var: tek bir sayı yanıltıcı olur, %95 güven aralığını da raporla
        out["custom_acc_ci95"] = wilson_ci(correct, int(cm.sum()))
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson %95 güven aralığı: n küçükken oranın ne kadar belirsiz olduğunu gösterir."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(center - half), 4), round(float(center + half), 4)]


def choose_threshold(y_val, pred_val, conf_val) -> tuple[float, dict]:
    """Val setinde (in_scope_acc + oos_recall) / 2'yi en büyük yapan eşiği seçer.
    İki metriğin ortalaması: biri artarken diğeri düşer (eşik yükselirse daha çok mesaj human agent'a gider)."""
    best_t, best_score, best_m = 0.0, -1.0, None
    for t in np.round(np.arange(0.0, 1.0, 0.01), 2):
        m = compute_metrics(y_val, decide(pred_val, conf_val, t))
        score = (m["in_scope_acc"] + m["oos_recall"]) / 2
        if score > best_score:
            best_t, best_score, best_m = float(t), score, m
    return best_t, best_m


# Eşiklerin alt sınırı. Ödev: "Low-confidence ... messages go to human agent". Alt sınır olmadan beyaz
# eşya eşiği val'de 0.00–0.08 çıkıyordu, yani düşük güven kontrolü fiilen kapanıyordu (val setinde
# "beyaz eşyaya benzeyen kapsam dışı" örnek olmadığı için optimizasyon bunun cezasını görmüyor).
# 0.3 val'de denendi (0 / 0.3 / 0.4 / 0.5): mBERT ve FFN'de kayıp yok, baseline custom_acc %90 → %85.
MIN_THRESHOLD = 0.3


def calibrate_thresholds(y_val, pred_val, conf_val, source_val, step: float = 0.02) -> tuple[dict, dict]:
    """İki eşiği VAL setinde birlikte seçer. Hedef: (in_scope_acc + oos_recall + custom_acc) / 3.
    Üç metriğin ortalaması: CLINC başarısı, kapsam dışını yakalama ve kendi alanımızdaki başarı eşit ağırlıkta.
    Her iki eşik de en az MIN_THRESHOLD olur."""
    grid = np.round(np.arange(MIN_THRESHOLD, 1.0, step), 2)
    best = None
    for t_clinc in grid:
        for t_custom in grid:
            th = {"clinc": float(t_clinc), "custom": float(t_custom)}
            m = compute_metrics(y_val, decide(pred_val, conf_val, th), source_val)
            score = (m["in_scope_acc"] + m["oos_recall"] + m["custom_acc"]) / 3
            if best is None or score > best[0]:
                best = (score, th, m)
    return best[1], best[2]


def save_metrics(model_name: str, metrics: dict, extra: dict | None = None) -> Path:
    """results/metrics_<model>.json. Test setinin parmak izi de yazılır (aynı test seti kanıtı)."""
    RESULTS.mkdir(exist_ok=True)
    payload = {"model": model_name, "test_sha256": split_sha256("test"), **metrics, **(extra or {})}
    path = RESULTS / f"metrics_{model_name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
