"""Model 3 · C · Transformer (öğretmen): mBERT, intent sınıflandırma için fine-tune.

Çalıştırma:  python -m src.train_teacher
Çıktılar:    models/teacher/ (ağırlıklar + tokenizer + router_config.json)
Sonra:       python -m src.calibrate teacher   (iki eşik + test değerlendirmesi)

Model: bert-base-multilingual-cased. Çok dilli (Türkçe dahil) önceden eğitilmiş BERT. Türkçe ve İngilizce
mesajları aynı modelle anlayabilmek için çok dilli bir model seçtik. Üzerine 156 sınıflı bir
sınıflandırma katmanı (task head) eklenir: [CLS] vektörü → Linear → 156 skor.
En iyi epoch VAL setinde seçilir. Bu dosya test setini hiç okumaz.
"""
from src.evaluate import ROOT, load_split
from src.hf_model import train_classifier

MODEL_NAME = "bert-base-multilingual-cased"
OUT = ROOT / "models" / "teacher"


def main() -> None:
    train, val = load_split("train"), load_split("val")
    train_classifier(MODEL_NAME, OUT, train, val, epochs=5, lr=5e-5, batch_size=32)
    print("Kaydedildi:", OUT)


if __name__ == "__main__":
    main()
