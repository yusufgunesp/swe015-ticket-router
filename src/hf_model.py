"""C · Transformer: Hugging Face modelleri için ortak eğitim ve tahmin kodu.

Hem öğretmen (mBERT) hem öğrenci (DistilmBERT, distillation ile) bu dosyadaki train_classifier()
ile eğitilir. Tahmin için HFPredictor kullanılır (benchmark ve uygulama).

Bellek notu (GTX 1650 Ti, 4 GB): Kelime gömme tablosu DONDURULUR. Bizim eğitim çıktımızda ölçülen:
mBERT 178M parametre, eğitilen 86M; DistilmBERT 135M parametre, eğitilen 44M.
  - Eğitilecek parametre sayısı yarıya iner, model 4 GB'a sığar.
  - Önceden eğitilmiş çok dilli gömmeler korunur. Türkçe/İngilizce aynı anlam uzayında kalır.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.evaluate import choose_threshold

MAX_LEN = 64  # CLINC cümleleri kısa. Train'deki en uzun cümle 64 token'ın altında.


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HFPredictor:
    """Kaydedilmiş bir sınıflandırıcıyı yükler. Tahmin: en olası niyet + softmax olasılığı (güven)."""

    def __init__(self, path: Path, device: str = "cpu", model=None):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = model if model is not None else AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(device).eval()
        self.device = device
        self.labels = np.array([self.model.config.id2label[i] for i in range(self.model.config.num_labels)],
                               dtype=object)
        cfg = Path(path) / "router_config.json"
        self.threshold = json.loads(cfg.read_text())["threshold"] if cfg.exists() else None

    @classmethod
    def from_loaded(cls, tokenizer, model, labels, device):
        """Eğitim sırasında, bellekteki modelle val tahmini yapmak için (diske yazmadan)."""
        obj = cls.__new__(cls)
        obj.tokenizer, obj.model, obj.device = tokenizer, model.eval(), device
        obj.labels, obj.threshold = np.array(labels, dtype=object), None
        return obj

    @torch.no_grad()
    def logits(self, texts, batch_size: int = 128) -> np.ndarray:
        out = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                 max_length=MAX_LEN, return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                out.append(self.model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out)

    def predict(self, texts, batch_size: int = 128):
        proba = torch.softmax(torch.from_numpy(self.logits(texts, batch_size)), dim=1).numpy()
        idx = proba.argmax(axis=1)
        return self.labels[idx], proba[np.arange(len(idx)), idx]


def train_classifier(model_name: str, out_dir: Path, train, val, *, epochs: int, lr: float,
                     batch_size: int = 32, seed: int = 42,
                     teacher_logits: np.ndarray | None = None, T: float = 2.0, alpha: float = 0.5,
                     teacher_labels: list | None = None) -> dict:
    """Fine-tune. teacher_logits verilirse knowledge distillation yapılır:
        kayıp = alpha · T² · KL( softmax(öğretmen/T) || softmax(öğrenci/T) )  +  (1 − alpha) · CE(öğrenci, gerçek etiket)
    T (sıcaklık) olasılıkları yumuşatır. Öğrenci sadece doğru sınıfı değil, öğretmenin
    "hangi sınıflar birbirine benziyor" bilgisini de öğrenir.
    En iyi epoch ve OOS eşiği VAL setinde seçilir."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = teacher_labels or sorted(train["intent"].unique())
    lab2id = {l: i for i, l in enumerate(labels)}

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)}, label2id=lab2id)
    model.get_input_embeddings().weight.requires_grad = False   # kelime gömmelerini dondur
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"{model_name}: {n_all / 1e6:.0f}M parametre, eğitilen {n_train / 1e6:.0f}M | cihaz: {device}")

    enc = tok(train["text_norm"].tolist(), padding="max_length", truncation=True,
              max_length=MAX_LEN, return_tensors="pt")
    y = torch.tensor(train["intent"].map(lab2id).values)
    t_logits = torch.from_numpy(teacher_logits).float() if teacher_logits is not None else None

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = epochs * ((len(y) + batch_size - 1) // batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best = {"score": -1}
    for ep in range(1, epochs + 1):
        model.train()
        perm, total, t0 = torch.randperm(len(y)), 0.0, time.time()
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            # Dinamik kırpma: batch'teki en uzun cümle kadar token tut. Cümleler ~15 token,
            # 64'e kadar dolgu yapmak hesaplamanın çoğunu boşa harcıyordu. Dolgu sağda
            # olduğu ve maskelendiği için sonuç değişmez, sadece hızlanır.
            L = int(enc["attention_mask"][b].sum(1).max())
            inputs = {k: v[b, :L].to(device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**inputs).logits
            logits = logits.float()
            loss = F.cross_entropy(logits, y[b].to(device))
            if t_logits is not None:
                kd = F.kl_div(F.log_softmax(logits / T, dim=1), F.softmax(t_logits[b].to(device) / T, dim=1),
                              reduction="batchmean") * T * T
                loss = alpha * kd + (1 - alpha) * loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
            step = i // batch_size + 1
            if step % 100 == 0:
                print(f"  epoch {ep} adım {step}/{(len(y) + batch_size - 1) // batch_size} | "
                      f"{time.time() - t0:.0f} sn", flush=True)

        # Val değerlendirmesi: uygulama ve test ile AYNI tahmin kodu (HFPredictor.predict)
        p, c = HFPredictor.from_loaded(tok, model, labels, device).predict(val["text_norm"].tolist())
        thr, m = choose_threshold(val["intent"], p, c)
        score = (m["in_scope_acc"] + m["oos_recall"]) / 2
        print(f"epoch {ep} | loss {total / len(y):.4f} | val in_scope_acc={m['in_scope_acc']:.4f} "
              f"oos_recall={m['oos_recall']:.4f} (eşik={thr:.2f}) | {time.time() - t0:.0f} sn", flush=True)
        if score > best["score"]:
            best = {"score": score, "epoch": ep, "threshold": thr, "val": m}
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            (out_dir / "router_config.json").write_text(json.dumps(
                {"threshold": thr, "epoch": ep, "lr": lr, "epochs": epochs, "batch_size": batch_size,
                 "max_len": MAX_LEN, "distill": t_logits is not None, "T": T, "alpha": alpha}, indent=2))
    print(f"Seçilen epoch: {best['epoch']}, eşik: {best['threshold']}")
    return best
