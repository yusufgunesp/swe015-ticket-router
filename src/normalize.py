"""B · Klasik NLP: metin normalizasyonu.

Tüm modeller AYNI normalize edilmiş metni görür. Uygulamada da aynı fonksiyon çalışır.

Adımlar (sırasıyla):
  1. Model kodu çıkarma (regex):           "DW-6534" → kaydedilir, metinde "modelcode" olur
  2. Küçük harfe çevirme (Türkçe "İ" dahil)
  3. Tekrar eden harfleri kısaltma (regex): "çoook" → "çook"
  4. Noktalama temizleme (regex)
  5. Ürün adı yazım düzeltme (minimum edit distance): "dishwaser" → "dishwasher"

Örnek:
    >>> normalize("Bulasik makinsi su boşaltmıyor!!! model DW-6534")
    NormResult(text='bulaşık makinesi su boşaltmıyor modelcode', model_codes=['DW-6534'],
               corrections=[('bulasik', 'bulaşık'), ('makinsi', 'makinesi')])
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

KNOWN_WORDS_FILE = Path(__file__).parent / "resources" / "known_words.txt"

# --- 1. Model kodu: markanın ürün serisi ön eki + isteğe bağlı tire + 3-5 rakam + isteğe bağlı 1 harf
#        DW = dishwasher, WM = washing machine, RF = refrigerator   (DW-6534, wm1200, RF-450X)
# Ön ekler sabit, çünkü her "2-3 harf + rakam" kalıbına izin verince CLINC'teki uçuş numaraları
# ("flight DL123") da model kodu sanılıyordu (denetçi bulgusu).
MODEL_CODE_RE = re.compile(r"\b(DW|WM|RF)-?(\d{3,5})([A-Za-z]?)\b", re.IGNORECASE)

# --- 3. Aynı HARF 3+ kez tekrar ediyorsa 2'ye indir ("helloooo" → "helloo")
# [^\W\d] = rakam olmayan harf. Rakamlara uygulanırsa "$20000" → "$200" olur (denetçi bulgusu).
REPEAT_RE = re.compile(r"([^\W\d])\1{2,}")

# --- 4. Harf, rakam, boşluk ve kesme işareti dışındaki her şey boşluk olur
PUNCT_RE = re.compile(r"[^\w\s']")

# --- 5. Ürün sözlüğü (lexicon): doğru yazılışlar
PRODUCT_LEXICON = [
    # Türkçe
    "bulaşık", "çamaşır", "makinesi", "makinem", "makinemin", "makinesinin", "makinenin",
    "buzdolabı", "buzdolabım", "buzdolabımın", "buzdolabının", "buzdolabında",
    "dondurucu", "garanti", "garantisi", "teknisyen", "randevu", "randevumu", "servis",
    # İngilizce
    "dishwasher", "washing", "machine", "washer", "refrigerator", "fridge", "freezer",
    "warranty", "technician", "appointment",
]

# Türkçe karakterleri ASCII'ye katlama: "bulasik" ile "bulaşık" karşılaştırılabilsin diye
_FOLD = str.maketrans("çğıöşüâî", "cgiosuai")


def fold(s: str) -> str:
    return s.translate(_FOLD)


def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance (Levenshtein): ekleme = silme = değiştirme = 1 maliyet.
    Dinamik programlama: D[i][j] = a[:i] ile b[:j] arasındaki en küçük maliyet."""
    prev = list(range(len(b) + 1))           # D[0][j] = j  (boş stringten j ekleme)
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)             # D[i][0] = i  (i silme)
        for j in range(1, len(b) + 1):
            sub = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,        # silme
                         cur[j - 1] + 1,     # ekleme
                         prev[j - 1] + sub)  # değiştirme (ya da eşleşme)
        prev = cur
    return prev[-1]


@lru_cache(maxsize=1)
def known_words() -> frozenset:
    """CLINC eğitim setinden çıkarılan doğru İngilizce kelimeler (src/data.py üretir)."""
    if KNOWN_WORDS_FILE.exists():
        return frozenset(KNOWN_WORDS_FILE.read_text(encoding="utf-8").split())
    return frozenset()


_FOLDED_LEXICON = {fold(w): w for w in PRODUCT_LEXICON}
_EDIT_TARGETS = {f: w for f, w in _FOLDED_LEXICON.items() if len(f) >= 7}


@lru_cache(maxsize=50_000)
def correct_word(word: str) -> str:
    """Bir kelimeyi ürün sözlüğündeki en yakın kelimeye düzeltir. Kurallar:
    - 4 harften kısa kelimelere dokunulmaz
    - Zaten sözlükte ya da bilinen kelimelerde olan kelime değişmez ("making" → "makine" olmasın)
    - Sözlük kelimesiyle BAŞLAYAN çekimli halleri değiştirmez ("garantili", "machines", "fridge's")
    - En fazla 1 mesafe, ve sadece 7+ harfli sözlük kelimelerine (Türkçe karakterler katlandıktan sonra)
      Val setinde denedik: 2 mesafeye ya da kısa hedeflere izin verince "watching"→"washing",
      "vaccine"→"machine", "washed"→"washer", "serves"→"servis" gibi hatalı düzeltmeler oluyordu.
    """
    if len(word) < 4 or word in PRODUCT_LEXICON or word in known_words():
        return word
    f = fold(word)
    if f in _FOLDED_LEXICON:                    # "bulasik" → "bulaşık" (sadece Türkçe karakter eksik)
        return _FOLDED_LEXICON[f]
    if any(f.startswith(f_lex) for f_lex in _FOLDED_LEXICON if len(f_lex) >= 5):
        return word                             # çekimli hal: doğru yazılmış, dokunma
    max_d = 1
    best, best_d = word, max_d + 1
    for f_lex, lex in _EDIT_TARGETS.items():
        if abs(len(f_lex) - len(f)) > max_d:    # uzunluk farkı zaten büyükse hesaplama
            continue
        d = edit_distance(f, f_lex)
        if d < best_d:
            best, best_d = lex, d
    return best if best_d <= max_d else word


@dataclass
class NormResult:
    text: str
    model_codes: list = field(default_factory=list)
    corrections: list = field(default_factory=list)


def normalize(text: str, correct: bool = True) -> NormResult:
    """correct=False: 5. adım (yazım düzeltme) atlanır. Sadece ablasyon ölçümü için (benchmark.py)."""
    # 1. Model kodlarını bul, standart biçimde kaydet (DW-6534), metinde tek bir jetonla değiştir
    codes = [f"{a.upper()}-{n}{s.upper()}" for a, n, s in MODEL_CODE_RE.findall(text)]
    text = MODEL_CODE_RE.sub(" modelcode ", text)
    # 2. Küçük harf. Python "İ".lower() → "i̇" (noktalı) verir, bu yüzden önce elle çeviriyoruz
    text = text.replace("İ", "i").replace("I", "ı" if _looks_turkish(text) else "i").lower()
    # 3. Tekrar eden harfler
    text = REPEAT_RE.sub(r"\1\1", text)
    # 4. Noktalama
    text = PUNCT_RE.sub(" ", text)
    # 5. Yazım düzeltme (kelime kelime)
    words, corrections = [], []
    for w in text.split():
        c = correct_word(w) if correct else w
        if c != w:
            corrections.append((w, c))
        words.append(c)
    return NormResult(" ".join(words), codes, corrections)


_TR_CHARS = set("çğıöşüÇĞİÖŞÜ")


def _looks_turkish(text: str) -> bool:
    """Büyük 'I' harfinin Türkçede 'ı', İngilizcede 'i' olması için kaba dil tahmini."""
    return any(ch in _TR_CHARS for ch in text)


def normalize_text(text: str) -> str:
    """Modellerin kullandığı kısa yol: sadece normalize edilmiş metni döndürür."""
    return normalize(text).text


if __name__ == "__main__":
    for s in ["Bulasik makinsi su boşaltmıyor!!! model DW-6534",
              "dishwaser not starting, model dw6534",
              "my refridgerator is making a loud noiseeeee",
              "camasir makinasi sikmiyor WM-1200",
              "I need to transfer money to my savings account"]:
        print(s, "\n  →", normalize(s))
