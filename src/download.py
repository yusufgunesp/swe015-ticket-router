"""Model ağırlıklarını Hugging Face Hub'dan indirir (R7: büyük dosyalar repoda değil, Hub'da).

app.py başlarken ensure_models() çağırır. models/ klasöründe eksik model varsa Hub'daki
HF_REPO_ID deposundan indirilir. Yerelde eğitilmiş modeller varsa indirme yapılmaz.
"""
import os

from src.predictors import MODEL_DIRS, MODELS

# Ağırlıkların yüklendiği public Hugging Face model deposu (ortam değişkeniyle değiştirilebilir)
HF_REPO_ID = os.environ.get("SWE015_HF_REPO", "yusufinfo/swe015-ticket-router-models")


def missing_models() -> list[str]:
    return [name for name, path in MODEL_DIRS.items() if not (path / "thresholds.json").exists()]


def ensure_models() -> None:
    missing = missing_models()
    if not missing:
        return
    if not HF_REPO_ID:
        raise SystemExit(f"Eksik modeller: {missing}. HF_REPO_ID ayarlı değil, indirilemiyor.")
    from huggingface_hub import snapshot_download
    print(f"Modeller Hugging Face Hub'dan indiriliyor ({HF_REPO_ID}): {missing}")
    snapshot_download(repo_id=HF_REPO_ID, local_dir=MODELS,
                      allow_patterns=[f"{m}/*" for m in missing])
    still = missing_models()
    if still:
        raise SystemExit(f"İndirme sonrası hâlâ eksik: {still}")
