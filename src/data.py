"""Veri hazırlama: CLINC150 (plus) + kendi beyaz eşya cümlelerimiz.

Çalıştırma:  python -m src.data
Çıktılar:
    data/processed/{train,val,test}.csv   (sütunlar: text, intent, source, lang)
    data/processed/SPLIT_INFO.json        (her bölmenin SHA-256 parmak izi)
    src/resources/known_words.txt         (yazım düzeltmede "zaten doğru" sayılan kelimeler)

ÖNEMLİ (R3): Bölmeler BİR KEZ, sabit seed ile üretilir. Tüm modeller aynı
data/processed/test.csv üzerinde değerlendirilir ve her sonuç dosyasına test.csv'nin
parmak izi yazılır. Böylece "aynı test seti" kanıtlanabilir.
Test seti eğitimde ve eşik ayarında ASLA kullanılmaz (eşik val ile seçilir).
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CUSTOM_CSV = ROOT / "data" / "custom" / "appliance_tr_en.csv"
OUT_DIR = ROOT / "data" / "processed"
KNOWN_WORDS = ROOT / "src" / "resources" / "known_words.txt"
SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_clinc() -> dict[str, pd.DataFrame]:
    """CLINC150 'plus' sürümü: 150 niyet + 'oos' (kapsam dışı) sınıfı. Resmi bölmeler aynen kullanılır."""
    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    out = {}
    for split_hf, split_ours in [("train", "train"), ("validation", "val"), ("test", "test")]:
        df = ds[split_hf].to_pandas()
        df["intent"] = df["intent"].map(lambda i: names[i])
        df["source"] = "clinc150"
        df["lang"] = "en"
        out[split_ours] = df[["text", "intent", "source", "lang"]]
    return out


def load_custom() -> dict[str, pd.DataFrame]:
    """Kendi cümlelerimiz: %60 train / %20 val / %20 test.
    Bölme niyet + dil birlikte korunarak yapılır, yani her niyetin test kısmında hem TR hem EN cümle olur."""
    if not CUSTOM_CSV.exists():
        # Denetçi uyarısı: cümleler sonradan eklenirse test seti değişir → modeller farklı
        # test setlerinde ölçülmüş olur (R3). Bu yüzden dosya yoksa hiç bölme üretmiyoruz.
        raise FileNotFoundError(f"{CUSTOM_CSV} bulunamadı. Bölmeler bu dosya olmadan üretilmez.")
    df = pd.read_csv(CUSTOM_CSV).dropna(subset=["text", "intent"])
    df["source"] = "custom"
    df = df[["text", "intent", "source", "lang"]]
    strata = df["intent"] + "_" + df["lang"]
    train, rest = train_test_split(df, test_size=0.4, stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=strata[rest.index], random_state=SEED)
    return {"train": train, "val": val, "test": test}


def build_known_words(train_clinc: pd.DataFrame) -> None:
    """Yazım düzeltme için 'bilinen kelimeler' listesi: CLINC eğitim setinde en az 2 kez geçen kelimeler.
    Sadece TRAIN kullanılır (test sızıntısı olmasın). Bu listedeki bir kelime 'yanlış yazılmış' sayılmaz."""
    counts = Counter(w for t in train_clinc["text"] for w in re.findall(r"[a-z']+", t.lower()))
    words = sorted(w for w, c in counts.items() if c >= 2)
    KNOWN_WORDS.parent.mkdir(parents=True, exist_ok=True)
    KNOWN_WORDS.write_text("\n".join(words), encoding="utf-8")
    print(f"known_words.txt: {len(words)} kelime")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    custom = load_custom()  # önce bu: dosya yoksa indirme yapmadan dur
    clinc = load_clinc()
    info = {"seed": SEED, "splits": {}}
    for split in ["train", "val", "test"]:
        df = pd.concat([clinc[split], custom[split]], ignore_index=True)
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # karıştır
        path = OUT_DIR / f"{split}.csv"
        df.to_csv(path, index=False)
        info["splits"][split] = {
            "rows": len(df),
            "intents": int(df["intent"].nunique()),
            "oos": int((df["intent"] == "oos").sum()),
            "custom": int((df["source"] == "custom").sum()),
            "sha256": sha256(path),
        }
        s = info["splits"][split]
        print(f"{split:5s}: {s['rows']:6d} örnek | {s['intents']:3d} niyet | oos={s['oos']} | "
              f"custom={s['custom']} | sha256={s['sha256'][:12]}…")
    (OUT_DIR / "SPLIT_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    build_known_words(clinc["train"])


if __name__ == "__main__":
    main()
