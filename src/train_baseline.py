"""Model 1 · B · Klasik baseline: tf-idf + Logistic Regression.

Çalıştırma:  python -m src.train_baseline
Çıktılar:    models/baseline/model.joblib, models/baseline/config.json

Sonra:       python -m src.calibrate baseline   (iki eşik + test değerlendirmesi)

Özellikler (FeatureUnion ile yan yana birleştirilir):
  - kelime tf-idf (1-2 gram):        "su boşaltmıyor" gibi kelime öbekleri
  - karakter tf-idf (2-5 gram):      "boşaltm", "makin" gibi parçalar. Türkçe ekler ve yazım
                                     hatalarına karşı daha dayanıklı.
Sınıflandırıcı: çok sınıflı Logistic Regression (156 sınıf, "oos" dahil).
C (regularizasyon) değeri VAL setinde seçilir. Bu dosya test setini hiç okumaz.
"""
import json
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.evaluate import ROOT, choose_threshold, load_split

OUT = ROOT / "models" / "baseline"


def build(C: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features), ("lr", LogisticRegression(C=C, max_iter=3000))])


def predict(model: Pipeline, texts) -> tuple[np.ndarray, np.ndarray]:
    """En olası niyet ve onun olasılığı (güven skoru)."""
    proba = model.predict_proba(texts)
    idx = proba.argmax(axis=1)
    return model.classes_[idx], proba[np.arange(len(idx)), idx]


def main() -> None:
    train, val = load_split("train"), load_split("val")

    # 1) C'yi val setinde seç
    best = None
    for C in [1.0, 10.0, 30.0, 100.0]:
        t0 = time.time()
        model = build(C).fit(train["text_norm"], train["intent"])
        pred, conf = predict(model, val["text_norm"])
        thr, m = choose_threshold(val["intent"], pred, conf)
        score = (m["in_scope_acc"] + m["oos_recall"]) / 2
        print(f"C={C:5.1f} | val in_scope_acc={m['in_scope_acc']:.4f} oos_recall={m['oos_recall']:.4f} "
              f"(eşik={thr:.2f}) | {time.time() - t0:.0f} sn")
        if best is None or score > best[0]:
            best = (score, C, thr, model)
    _, C, thr, model = best
    print(f"Seçilen: C={C}, eşik={thr}")

    # 2) Kaydet. Test değerlendirmesi burada YAPILMAZ: eşik kalibrasyonu + test → src/calibrate.py
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")
    (OUT / "config.json").write_text(json.dumps({"C": C, "threshold": thr}, indent=2))
    print("Kaydedildi:", OUT)


if __name__ == "__main__":
    main()
