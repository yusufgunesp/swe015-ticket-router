"""Eşik kalibrasyonu + TEST değerlendirmesi. Test seti, model doğruluğunu ölçmek için SADECE burada okunur.

Çalıştırma:  python -m src.calibrate                     (eğitilmiş tüm modeller)
             python -m src.calibrate teacher student     (sadece seçilenler)

Her model için:
  1. VAL setinde tahmin → iki eşik seçilir (evaluate.calibrate_thresholds) → models/<m>/thresholds.json
  2. TEST setinde tahmin → iki eşikli karar kuralıyla metrikler → results/metrics_<m>.json
     Karşılaştırma için eğitimde seçilen tek eşikli sonuç da aynı dosyaya yazılır ("single_threshold_metrics").
Tahminler CPU'da yapılır (uygulama da CPU'da çalışacak).
"""
import json
import sys
import time

from src.evaluate import MIN_THRESHOLD, calibrate_thresholds, compute_metrics, decide, load_split, save_metrics
from src.predictors import MODEL_DIRS, load_predictor


def calibrate(name: str) -> dict:
    val, test = load_split("val"), load_split("test")
    t0 = time.time()
    pred = load_predictor(name, device="cpu")

    pv, cv = pred.predict(val["text_norm"].tolist())
    th, mval = calibrate_thresholds(val["intent"], pv, cv, val["source"])
    (pred.path / "thresholds.json").write_text(json.dumps(
        {"thresholds": th, "objective": "(in_scope_acc + oos_recall + custom_acc) / 3 on val",
         "min_threshold": MIN_THRESHOLD,
         "val_metrics": mval}, indent=2))

    pt, ct = pred.predict(test["text_norm"].tolist())
    metrics = compute_metrics(test["intent"], decide(pt, ct, th), test["source"])
    single = compute_metrics(test["intent"], decide(pt, ct, pred.single_threshold), test["source"])
    save_metrics(name, metrics, {
        "thresholds": th, "val_metrics": mval,
        "single_threshold": pred.single_threshold, "single_threshold_metrics": single})

    print(f"{name:8s} | eşikler clinc={th['clinc']:.2f} custom={th['custom']:.2f} | "
          f"TEST in={metrics['in_scope_acc']:.4f} oos={metrics['oos_recall']:.4f} "
          f"custom={metrics['custom_acc']:.4f} {metrics['custom_acc_ci95']} | "
          f"(tek eşik {pred.single_threshold:.2f}: in={single['in_scope_acc']:.4f} "
          f"oos={single['oos_recall']:.4f} custom={single['custom_acc']:.4f}) | {time.time() - t0:.0f} sn",
          flush=True)
    return metrics


def main() -> None:
    names = sys.argv[1:] or [n for n, p in MODEL_DIRS.items() if p.exists()]
    for name in names:
        calibrate(name)


if __name__ == "__main__":
    main()
