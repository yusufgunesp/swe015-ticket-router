"""Model 2 · A · Sinir ağı: kendi eğittiğimiz fastText gömmelerinin ortalaması → feedforward ağ (SGD + backprop).

Çalıştırma:  python -m src.train_ffn
Çıktılar:    models/ffn/fasttext.model, models/ffn/ffn.pt, models/ffn/config.json
Sonra:       python -m src.calibrate ffn   (iki eşik + test değerlendirmesi)

Adımlar:
  1. fastText (gensim) SADECE train cümleleri üzerinde eğitilir. fastText kelimeyi karakter
     parçalarına (3-5 gram) böler, bu sayede "boşaltmıyor" gibi görülmemiş kelimelere ve yazım
     hatalarına da vektör üretebilir.
  2. Cümle vektörü = cümledeki kelime vektörlerinin ortalaması (100 boyut).
  3. Feedforward ağ: 100 → 256 (ReLU, dropout) → 156 sınıf (softmax).
     Kayıp: cross-entropy. Optimizasyon: torch.optim.SGD (momentum'lu). Gradyanlar backprop ile
     hesaplanır (loss.backward()).
  4. En iyi epoch VAL setinde seçilir. Bu dosya test setini hiç okumaz.
"""
import json
import random
import time

import numpy as np
import torch
from gensim.models import FastText
from torch import nn

from src.evaluate import ROOT, choose_threshold, load_split

OUT = ROOT / "models" / "ffn"
SEED = 42
EMB_DIM = 100


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FFN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),   # gizli katman: W1·x + b1
            nn.ReLU(),
            nn.Dropout(dropout),         # eğitimde nöronların %30'unu rastgele kapat (ezberi azaltır)
            nn.Linear(hidden, n_classes) # çıkış: her sınıf için bir skor (logit). Softmax kayıpta.
        )

    def forward(self, x):
        return self.net(x)


def sentence_vectors(ft: FastText, texts) -> np.ndarray:
    """Her cümle için kelime vektörlerinin ortalaması."""
    out = np.zeros((len(texts), ft.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        words = t.split()
        if words:
            out[i] = np.mean([ft.wv[w] for w in words], axis=0)
    return out


class FFNPredictor:
    """Eğitilmiş modeli yükleyip tahmin yapan sınıf (benchmark ve uygulama bunu kullanır)."""

    def __init__(self, path=OUT):
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        self.labels = np.array(cfg["labels"], dtype=object)
        self.threshold = cfg["threshold"]
        self.mean, self.std = np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32)
        self.ft = FastText.load(str(path / "fasttext.model"))
        self.model = FFN(EMB_DIM, cfg["hidden"], len(self.labels))
        self.model.load_state_dict(torch.load(path / "ffn.pt", map_location="cpu"))
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts_norm):
        x = (sentence_vectors(self.ft, list(texts_norm)) - self.mean) / self.std
        proba = torch.softmax(self.model(torch.from_numpy(x)), dim=1).numpy()
        idx = proba.argmax(axis=1)
        return self.labels[idx], proba[np.arange(len(idx)), idx]


def main() -> None:
    set_seed()
    train, val = load_split("train"), load_split("val")

    # 1) fastText: sadece train cümleleri
    t0 = time.time()
    sentences = [t.split() for t in train["text_norm"]]
    ft = FastText(sentences, vector_size=EMB_DIM, window=5, min_count=1, sg=1, epochs=20,
                  min_n=3, max_n=5, bucket=100_000, seed=SEED, workers=1)
    print(f"fastText eğitildi: {len(ft.wv)} kelime, {time.time() - t0:.0f} sn")

    # 2) Cümle vektörleri + standartlaştırma (ortalama/std SADECE train'den)
    Xtr = sentence_vectors(ft, train["text_norm"].tolist())
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    norm = lambda X: (X - mean) / std
    Xtr, Xva = norm(Xtr), norm(sentence_vectors(ft, val["text_norm"].tolist()))

    labels = np.array(sorted(train["intent"].unique()), dtype=object)
    lab2id = {l: i for i, l in enumerate(labels)}
    ytr = torch.tensor(train["intent"].map(lab2id).values)
    Xtr_t, Xva_t = torch.from_numpy(Xtr), torch.from_numpy(Xva)

    # 3) FFN + SGD
    hidden, epochs, batch = 256, 80, 64
    model = FFN(EMB_DIM, hidden, len(labels))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()  # softmax + negatif log-olabilirlik

    best = {"score": -1}
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr))
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), ytr[b])
            loss.backward()   # backprop: her ağırlığın gradyanı
            opt.step()        # SGD: w ← w − lr · gradyan (+ momentum)
            total += loss.item() * len(b)
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                proba = torch.softmax(model(Xva_t), dim=1).numpy()
            pred, conf = labels[proba.argmax(1)], proba.max(1)
            thr, m = choose_threshold(val["intent"], pred, conf)
            score = (m["in_scope_acc"] + m["oos_recall"]) / 2
            print(f"epoch {ep:3d} | loss {total / len(ytr):.4f} | val in_scope_acc={m['in_scope_acc']:.4f} "
                  f"oos_recall={m['oos_recall']:.4f} (eşik={thr:.2f})")
            if score > best["score"]:
                best = {"score": score, "epoch": ep, "threshold": thr,
                        "state": {k: v.clone() for k, v in model.state_dict().items()}}
    print(f"Seçilen epoch: {best['epoch']}, eşik: {best['threshold']}")

    # 4) Kaydet. Test değerlendirmesi burada YAPILMAZ: eşik kalibrasyonu + test → src/calibrate.py
    OUT.mkdir(parents=True, exist_ok=True)
    ft.save(str(OUT / "fasttext.model"))
    torch.save(best["state"], OUT / "ffn.pt")
    cfg = {"labels": labels.tolist(), "threshold": best["threshold"], "hidden": hidden,
           "epoch": best["epoch"], "mean": mean.tolist(), "std": std.tolist(),
           "optimizer": "SGD(lr=0.05, momentum=0.9, weight_decay=1e-4)"}
    (OUT / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    print("Kaydedildi:", OUT)


if __name__ == "__main__":
    main()
