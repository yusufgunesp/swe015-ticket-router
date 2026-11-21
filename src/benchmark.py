"""R4 · Sonuç tablosu: her model için görev metrikleri, model boyutu (MB) ve gecikme (ms, ≥100 çalıştırma, CPU).

Çalıştırma:  python -m src.benchmark          (önce src.calibrate çalışmış olmalı)
Çıktılar:
    results/results.csv, results/results.md   → sunumdaki tablo (sayılar elle yazılmaz, buradan alınır)
    results/hardware.json                      → ölçüm yapılan donanım
    results/errors_<model>.csv                 → yanlış tahmin edilen test cümleleri (hata örnekleri için)

Tanımlar:
    Accuracy            : 5.560 test cümlesinin tamamında doğru karar oranı (iki eşikli kural, "oos" dahil)
    Out-of-scope recall : gerçek "oos" cümlelerinden kaçı human agent'a gitti
    In-scope accuracy   : "oos" olmayan cümlelerde doğru niyet oranı (4.500 CLINC + 60 beyaz eşya)
    Appliance acc.      : 60 beyaz eşya test cümlesi (TR ve EN ayrıca)
    Model size (MB)     : çıkarım için gereken tüm dosyalar (ağırlıklar + tokenizer/vektörizer),
                          sadece küçük JSON ayar dosyaları hariç
    Latency (ms)        : TEK mesaj (batch=1) için normalizasyon + tokenizasyon + model + softmax süresi.
                          Sabit 200 test cümlesi, her model için aynı: 60 beyaz eşya (40 TR + 20 EN) +
                          test sırasındaki ilk 140 CLINC cümlesi. 10 ısınma çalıştırmasından sonra 200
                          çalıştırmanın ortalaması. CPU, PyTorch ve ONNX Runtime aynı thread sayısıyla.
    Appliance acc. (no spelling corr.) : ablasyon. Aynı model ve eşikler, ama yazım düzeltme kapalı.
Görev metrikleri calibrate.py'nin kaydettiği sonuçlarla karşılaştırılır. Uyuşmazlık varsa hata verilir.
"""
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluate import RESULTS, compute_metrics, decide, load_split, split_sha256
from src.normalize import normalize, normalize_text
from src.predictors import MODEL_DIRS, load_predictor

ROWS = [  # (anahtar, tablodaki ad, merdivendeki rol)
    ("baseline", "tf-idf + Logistic Regression", "Classical baseline (B)"),
    ("ffn", "fastText avg. + FFN (SGD)", "Neural network (A)"),
    ("teacher", "mBERT (bert-base-multilingual-cased)", "Transformer (C)"),
    ("student", "DistilmBERT, distilled from mBERT", "Distilled student (D)"),
    ("student_int8", "DistilmBERT INT8 (ONNX Runtime)", "Quantized Transformer (D), used in app"),
    ("teacher_int8", "mBERT INT8 (ONNX Runtime)", "Quantized teacher (extra row)"),
]
META_FILES = {"config.json", "thresholds.json", "router_config.json", "quantization.json",
              "tokenizer_config.json", "special_tokens_map.json"}
N_LAT, N_WARM = 200, 10


def hardware() -> dict:
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        import ctypes

        class MEM(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEM()
        m.dwLength = ctypes.sizeof(MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram = round(m.ullTotalPhys / 2**30, 1)
    else:
        ram = None
    import onnxruntime
    os_name = platform.platform()
    if sys.platform == "win32" and int(platform.version().split(".")[-1]) >= 22000:
        os_name = f"Windows 11 (build {platform.version()})"  # Python Windows 11'i "10" diye raporluyor
    return {"cpu": cpu, "ram_gb": ram, "os": os_name, "python": platform.python_version(),
            "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
            "torch_threads": torch.get_num_threads(), "device": "CPU (GPU not used for measurement)"}


def model_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file() and f.name not in META_FILES) / 2**20


def describe(name: str, pred) -> dict:
    """Hassasiyet ve parametre/katman sayısı: modelin kendisinden okunur (model kartından değil)."""
    if name == "baseline":
        lr = pred.model.named_steps["lr"]
        return {"precision": str(lr.coef_.dtype), "params_M": lr.coef_.size / 1e6 + lr.intercept_.size / 1e6,
                "layers": "-"}
    if name == "ffn":
        n = sum(p.numel() for p in pred.model.parameters())
        dt = next(pred.model.parameters()).dtype
        return {"precision": f"{str(dt).replace('torch.', '')} (FFN) + {pred.ft.wv.vectors.dtype} (fastText)",
                "params_M": n / 1e6, "layers": "1 hidden"}
    if name.endswith("_int8"):
        # Hassasiyet ONNX grafiğinin kendisinden okunur: ağırlık (initializer) tipleri ve INT8 düğümleri
        import onnx
        from onnx import TensorProto
        graph = onnx.load_model_from_string((pred_path(name) / "model.onnx").read_bytes()).graph
        by_type = Counter()
        for init in graph.initializer:
            by_type[TensorProto.DataType.Name(init.data_type)] += int(np.prod(init.dims)) if init.dims else 1
        total = sum(by_type.values())
        top, ops = by_type.most_common(1)[0], Counter(n.op_type for n in graph.node)
        cfg = json.loads((pred_path(name) / "config.json").read_text())
        return {"precision": f"INT8 ({top[0].lower()} weights {100 * top[1] / total:.2f}%, dynamic activations)",
                "precision_detail": f"initializers {dict(by_type)}; nodes: {ops['DynamicQuantizeLinear']} "
                                    f"DynamicQuantizeLinear, {ops['MatMulInteger']} MatMulInteger, "
                                    f"{ops['MatMul']} FP32 MatMul (attention activations)",
                "params_M": total / 1e6, "layers": cfg.get("num_hidden_layers") or cfg.get("n_layers")}
    m = pred.model
    return {"precision": str(next(m.parameters()).dtype).replace("torch.", ""),
            "params_M": sum(p.numel() for p in m.parameters()) / 1e6,
            "layers": getattr(m.config, "num_hidden_layers", None) or getattr(m.config, "n_layers", None)}


def pred_path(name: str) -> Path:
    return MODEL_DIRS[name]


def latency_ms(pred, texts) -> dict:
    """Tek mesaj: ham metin → normalize → model → (niyet, güven)."""
    for t in texts[:N_WARM]:
        pred.predict([normalize_text(t)])
    times = []
    for t in texts[:N_LAT]:
        t0 = time.perf_counter()
        pred.predict([normalize_text(t)])
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return {"latency_ms": float(a.mean()), "latency_p50_ms": float(np.median(a)),
            "latency_p95_ms": float(np.percentile(a, 95)), "latency_runs": len(a)}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    test = load_split("test")
    hw = hardware()
    print("Donanım:", hw, flush=True)

    cust_idx = list(test.index[test["source"] == "custom"])
    clinc_idx = list(test.index[test["source"] != "custom"][:N_LAT - len(cust_idx)])
    lat_texts = test.loc[cust_idx + clinc_idx, "text"].tolist()
    custom_df = test[test["source"] == "custom"]
    custom_raw_norm = [normalize(t, correct=False).text for t in custom_df["text"]]
    n_changed = sum(a != b for a, b in zip(custom_df["text_norm"], custom_raw_norm))  # düzeltmenin değiştirdiği cümle
    clinc_df = test[test["source"] != "custom"]
    n_changed_clinc = sum(a != normalize(t, correct=False).text for a, t in zip(clinc_df["text_norm"], clinc_df["text"]))
    ins = (test["intent"] != "oos").values

    rows = []
    for name, label, role in ROWS:
        if not MODEL_DIRS[name].exists():
            print(f"[atlandı] {name}: model yok")
            continue
        pred = load_predictor(name, device="cpu")
        saved = json.loads((RESULTS / f"metrics_{name}.json").read_text(encoding="utf-8"))
        assert saved["test_sha256"] == split_sha256("test"), f"{name}: farklı test seti!"
        th = pred.thresholds
        assert th is not None, f"{name}: thresholds.json yok, önce calibrate.py çalıştırın"

        p, c = pred.predict(test["text_norm"].tolist())
        final = decide(p, c, th)
        m = compute_metrics(test["intent"], final, test["source"])
        for k in ["in_scope_acc", "oos_recall", "overall_acc", "custom_acc"]:
            assert abs(m[k] - saved[k]) < 1e-9, f"{name}: {k} calibrate sonucu ile uyuşmuyor"
        cust = (test["source"] == "custom").values
        by_lang = {lang: float((final[cust & (test["lang"] == lang).values] ==
                                test["intent"].values[cust & (test["lang"] == lang).values]).mean())
                   for lang in ["tr", "en"]}

        wrong = final != test["intent"].values
        pd.DataFrame({"text": test["text"][wrong], "true": test["intent"][wrong], "pred": final[wrong],
                      "model_top": p[wrong], "confidence": np.round(c[wrong], 4),
                      "source": test["source"][wrong], "lang": test["lang"][wrong]}
                     ).to_csv(RESULTS / f"errors_{name}.csv", index=False)

        if name.endswith("_int8"):  # ORT thread sayısı oturumun kendisinden okunur
            hw["onnxruntime_intra_op_threads"] = pred.session.get_session_options().intra_op_num_threads
        # Eşiksiz in-scope doğruluk: sadece en olası niyet. Eşik seçiminin etkisi olmadan modeli kıyaslamak için
        no_thr_in = float((p[ins] == test["intent"].values[ins]).mean())
        p_nc, c_nc = pred.predict(custom_raw_norm)  # ablasyon: yazım düzeltme kapalı
        no_corr = float((decide(p_nc, c_nc, th) == custom_df["intent"].values).mean())

        lat = latency_ms(pred, lat_texts)
        row = {"key": name, "Model": label, "Role": role, **describe(name, pred),
               "Accuracy": m["overall_acc"], "Out-of-scope recall": m["oos_recall"],
               "In-scope accuracy": m["in_scope_acc"], "In-scope acc. (no threshold)": no_thr_in,
               "Appliance acc.": m["custom_acc"],
               "Appliance acc. CI95": m["custom_acc_ci95"], "Appliance acc. TR": by_lang["tr"],
               "Appliance acc. EN": by_lang["en"], "Appliance acc. (no spelling corr.)": no_corr,
               "Model size (MB)": model_size_mb(MODEL_DIRS[name]),
               **lat, "thresholds": th, "test_sha256": saved["test_sha256"]}
        rows.append(row)
        print(f"{name:13s} acc={m['overall_acc']:.4f} oos={m['oos_recall']:.4f} "
              f"size={row['Model size (MB)']:.1f}MB lat={lat['latency_ms']:.1f}ms "
              f"(p95 {lat['latency_p95_ms']:.1f})", flush=True)

    (RESULTS / "hardware.json").write_text(json.dumps(hw, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "results.csv", index=False)
    pct = lambda x: f"{100 * x:.1f}%"
    md = ["| Model | Precision | Accuracy | Out-of-scope recall | In-scope acc. | Appliance acc. (TR / EN) "
          "| Model size (MB) | Latency (ms) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['Model']} | {r['precision']} | {pct(r['Accuracy'])} | {pct(r['Out-of-scope recall'])} | "
                  f"{pct(r['In-scope accuracy'])} | {pct(r['Appliance acc.'])} ({pct(r['Appliance acc. TR'])} / "
                  f"{pct(r['Appliance acc. EN'])}) | {r['Model size (MB)']:.1f} | {r['latency_ms']:.1f} |")
    md.append("")
    md.append(f"Hardware: {hw['cpu']}, {hw['ram_gb']} GB RAM, {hw['os']}; CPU only, batch size 1, "
              f"latency averaged over {N_LAT} runs (60 appliance TR/EN + 140 CLINC messages) after {N_WARM} "
              f"warm-up runs, CPU threads: PyTorch {hw['torch_threads']}, ONNX Runtime "
              f"{hw.get('onnxruntime_intra_op_threads')} (onnxruntime {hw['onnxruntime']}). "
              f"Test set: {len(test)} sentences (sha256 {split_sha256('test')[:12]}).")
    md.append("")
    md.append("In-scope acc. without threshold (top-1 intent only): "
              + "; ".join(f"{r['key']} {pct(r['In-scope acc. (no threshold)'])}" for r in rows))
    md.append("")
    md.append(f"Spelling-correction ablation (inference only; models were trained on corrected text). "
              f"Correction changes {n_changed} of {len(custom_df)} appliance test sentences and {n_changed_clinc} of "
              f"{len(clinc_df)} CLINC test sentences. Appliance acc. with → without correction: "
              + "; ".join(f"{r['key']} {pct(r['Appliance acc.'])} → {pct(r['Appliance acc. (no spelling corr.)'])}"
                          for r in rows))
    (RESULTS / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
