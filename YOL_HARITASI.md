# SWE015 Ödev – Yol Haritası (Use Case 2: Customer Support Ticket Router)

**Teslim:** 22 Kasım 2026 Pazar · **Notlama:** 0 / 50 / 100 (kontrol listesinin tamamı = 100)

> **KALDIĞIMIZ YER (20 Kas gecesi):** 6 model eğitildi, kalibre edildi ve ölçüldü (`results/results.md`). Router, uçtan uca değerlendirme ve `app.py` yazıldı. **app.py henüz tarayıcıda test edilmedi.**
>
> **Cumartesi sırası:**
> 1. Uygulamayı tarayıcıda test et.
> 2. Doğruluk denetimi: batch=1 düzeltmesi ve app.py.
> 3. Hugging Face hesabı ve model yükleme (giriş ve token işini sen yapacaksın).
> 4. GitHub reposu.
> 5. README.
> 6. Temiz ortam testi.
> 7. Mimari diyagram.
> 8. 12 slayt.
> 9. Gereksinim denetimi.
>
> **Pazar:** Video, linkler, PDF, commit (yazar adı o zaman sorulacak) ve teslim formu.

## 1. Uygulama ne yapacak?

Kullanıcı serbest metin yazar (TR veya EN) → metin normalize edilir → niyet (intent) tahmini ve güven skoru → yönlendirilen ekip gösterilir.

- Güven skoru eşiğin altındaysa ya da `oos` (kapsam dışı) ise → **"human agent"**
- Selamlaşma / sohbet ise → **ELIZA tarzı kural tabanlı cevap**
- 4 modelin tahmini **yan yana** gösterilir, asıl karar **quantize edilmiş Transformer** ile verilir (R5)

## 2. Veri

| Kaynak | İçerik | Neden |
|---|---|---|
| **CLINC150 (`plus` versiyonu)** | 150 niyet + `oos` sınıfı, İngilizce. Train 15.250 / Val 3.100 / Test 5.500 | Ödevin istediği **out-of-scope recall** ölçümü için hazır `oos` sınıfı var |
| **Kendi yazdığımız 300 cümle** | Beyaz eşya destek mesajları, 5 niyet × (40 TR + 20 EN). Örnek: "bulaşık makinem su boşaltmıyor, model DW-6534" | Senaryoya özgü alan + Türkçe. Bölme: 180 train / 60 val / 60 test (niyet + dil oranı korunarak) |

> **Veri sürümü notu (19 Kas):** İlk sürüm 100 cümleydi (eğitimde sadece 30 Türkçe cümle vardı). İlk baseline ve FFN sonuçlarında Türkçe kapsamının çok zayıf olduğunu gördük (FFN custom_acc %15). Bu yüzden her niyete 30 TR + 10 EN yeni cümle ekledik (toplam 300). Yeni cümleler eskilerin çevirisi değil. Bölmeler yeniden üretildi ve kilitlendi (test sha256: `7e8db122c096…`). **Tüm modeller bu yeni test setiyle yeniden eğitilip ölçüldü.** Eski sonuçlar kullanılmıyor. Cümleler hocanın izniyle AI yardımıyla yazıldı. AI kullanım slaytında belirtilecek.

- **Niyet → Ekip eşlemesi:** CLINC'in 10 alanı (banking, home, travel, small_talk …) ve bizim beyaz eşya niyetlerimiz ekiplere eşlenir.
- **Test seti:** Tüm modeller **aynı** sabit test seti üzerinde değerlendirilir (R3).

## 3. Teknik haritası (R2)

| Alan | Bileşen | Teknik |
|---|---|---|
| **B · Klasik NLP** | `normalize.py` | Regex normalizasyon (küçük harf, tekrar eden harf, noktalama), **model kodu çıkarma** (`DW-6534`) |
| | `normalize.py` | Yanlış yazılan ürün adlarını **minimum edit distance** ile düzeltme (`bulasik makinsi` → `bulaşık makinesi`) |
| | `eliza.py` | Sohbet / selamlaşma için **ELIZA tarzı** regex kuralları |
| | Model 1 | **tf-idf + Logistic Regression** baseline |
| **A · Sinir ağları** | Model 2 | Kendi eğittiğimiz **fastText/word2vec** gömmelerinin ortalaması → **feedforward ağ**, **SGD + backprop** (PyTorch) |
| **C · Transformer** | Model 3 (öğretmen) | **mBERT** (`bert-base-multilingual-cased`), intent sınıflandırma için fine-tune |
| **D · Verimli modeller** | Model 4 (öğrenci) | **Knowledge distillation** ile `distilbert-base-multilingual-cased` modeline aktarma |
| | Model 5 | Öğrencinin **dinamik INT8 quantization** hali (PyTorch veya ONNX Runtime). **Uygulamada bu çalışır** |

## 4. Sonuç tablosu (R4)

Sütun adları hocanın kelimeleriyle (İngilizce) yazılacak. Ek sütunların tanımı dipnotta verilecek.

| Model | Precision | Accuracy | Out-of-scope recall | In-scope acc. | Appliance acc. (TR/EN) | Model size (MB) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| tf-idf + LR (baseline) | float64 (doğrulanacak) | | | | | | |
| fastText + FFN (SGD) | FP32 | | | | | | |
| mBERT (Transformer) | FP32 | | | | | | |
| DistilmBERT (distilled student) | FP32 | | | | | | |
| **DistilmBERT INT8 (quantized Transformer, app)** | INT8 | | | | | | |
| *mBERT INT8 (sigorta satırı)* | INT8 | | | | | | |

- **Accuracy:** 5.560 test cümlesinin tamamında, iki eşikli karar kuralıyla.
- **Sigorta satırı:** R3 "**the** quantized Transformer" diyor. mBERT'in INT8 hali de ölçülüyor, böylece cümle iki şekilde de okunsa karşılanmış olur. UC2-D'ye göre asıl model distilled öğrencinin INT8 hali.
- **Latency:** batch=1, CPU, ≥100 çalıştırma ortalaması, ısınma sonrası. Thread sayısı yazılacak.

Donanım: Intel i5-10300H, 16 GB RAM (gecikme CPU'da ölçülür), eğitim GTX 1650 Ti 4 GB.
**Tüm sayılar `benchmark.py` çıktısından gelir. Elle sayı yazılmaz.**

## 5. Proje yapısı

```
LLM/
├── app.py                  # Gradio arayüzü (tek çalıştırma komutu)
├── requirements.txt
├── README.md
├── data/custom/appliance_tr_en.csv   # kendi yazdığımız cümleler
├── src/
│   ├── data.py             # veri yükleme + sabit bölmeler
│   ├── normalize.py        # regex, model kodu, edit distance
│   ├── eliza.py            # sohbet kuralları
│   ├── router.py           # niyet → ekip, eşik, human agent
│   ├── train_baseline.py
│   ├── train_ffn.py
│   ├── train_teacher.py
│   ├── distill_student.py
│   ├── quantize.py
│   └── benchmark.py        # accuracy, OOS recall, MB, ms → results/results.csv
├── models/                 # .gitignore (ağırlıklar Hugging Face Hub'a)
└── results/
```

## 6. Günlük plan

| Gün | Yapılacaklar | Çıktı |
|---|---|---|
| **Çar 18 Kas** | Sanal ortam + PyTorch (CUDA) kurulumu · CLINC150 indirme · ~100 beyaz eşya cümlesini yazma · `normalize.py`, `eliza.py` · tf-idf baseline | Baseline skoru |
| **Per 19 Kas** | fastText eğitimi + FFN (SGD) · mBERT öğretmen fine-tune (GPU) · OOS eşiğini validation setinde ayarlama | 3 model hazır |
| **Cum 20 Kas** | Distillation → öğrenci · INT8 quantization · `benchmark.py` (sonuç tablosu) · Gradio uygulaması | Çalışan uygulama + tablo |
| **Cmt 21 Kas** | Ağırlıkları Hugging Face'e yükleme · GitHub repo · README'yi temiz ortamda test · mimari diyagram · 8–12 slayt | Repo + sunum |
| **Paz 22 Kas** | 2 dk sesli video · linkleri gizli pencerede kontrol · teslim | Teslim ✅ |

## 7. Denetçi ajanlar (`.claude/agents/`)

| Denetçi | Ne kontrol eder? | Ne zaman çağrılır? |
|---|---|---|
| `gereksinim-denetcisi` | R1–R8 ve 0/50 kapıları, plana göre ilerleme | Her günün sonunda + sunum/video sonrası |
| `dogruluk-denetcisi` | Test sızıntısı, aynı test seti, metrik tanımları, gecikme/boyut ölçümü, quantize gerçekten INT8 mi, sayılar results.csv ile eşleşiyor mu | Her model eğitildikten sonra + sunum sonrası |
| `temiz-ortam-testcisi` | Sıfırdan README ile kurulum + tek komutla çalıştırma + örnek girdiler | Cumartesi (GitHub linkiyle) + README her değiştiğinde |

**Takvim:**
- **Çar sonu:** gereksinim
- **Per:** her modelden sonra doğruluk, gün sonunda gereksinim
- **Cum:** distillation ve quantization sonrası doğruluk, gün sonunda gereksinim
- **Cmt:** temiz ortam, sunum sonrası doğruluk ve gereksinim
- **Paz:** teslimden önce son gereksinim

Denetçiler dosya değiştirmez, sadece rapor verir. Düzeltmeleri birlikte yapar, her birinin nedenini konuşuruz.

## 8. Açık notlar (denetim bulguları, 18 Kas)

Düzeltilenler:
- Rakamlar kısaltılıyordu ("$20000" → "$200"). Artık sadece harfler kısaltılıyor.
- Uçuş numaraları model kodu sanılıyordu. Artık sadece DW/WM/RF ön ekli kodlar yakalanıyor.
- C ızgarası genişletildi (C=100 eklendi).
- custom_acc artık %95 güven aralığıyla birlikte raporlanıyor.

**Eşik yöntemi değişikliği (19 Kas):**
- **Sorun:** Tek ortak eşikte mBERT'in test custom_acc'si %16.7 çıktı. Val setinde teşhis edildi: model beyaz eşya cümlelerinin %90'ını doğru tahmin ediyor ama güveni ~0.6. Eşik (0.81) ise 3.100 CLINC cümlesine göre seçiliyor.
- **Çözüm:** İki ayrı eşik kullanılıyor: CLINC niyetleri için bir eşik, beyaz eşya niyetleri için bir eşik. İkisi val setinde birlikte seçiliyor. Hedef: (in_scope_acc + oos_recall + custom_acc) / 3. Yöntem 4 modelin hepsine aynen uygulanıyor.
- **Test setini artık sadece `src/calibrate.py` okuyor.** Eğitim script'leri test setine dokunmuyor.
- **Şeffaflık:** Tek eşikli sonuçlar da `results/metrics_*.json` içinde (`single_threshold_metrics`) saklanıyor. Sunumda yöntemin neden ve ne zaman değiştiği yazılacak.
- **Model seçimi** (C/epoch) eğitimde tek eşikli ölçütle yapıldı. Eşikler sonradan kalibre ediliyor.

Sonraki adımlarda yapılacaklar:
- [ ] **normalize.py kuralları DONDURULDU.** Bundan sonra test sonucuna bakıp kural değiştirilmez.
- [ ] **router.py:** `evaluate.decide()` ve modelin `thresholds.json` dosyasındaki iki eşik aynen kullanılacak. Ayrıca ELIZA ön filtresi dahil uçtan uca bir değerlendirme benchmark'a eklenecek. Sunumda "model metriği" ile "uygulama metriği" ayrı ayrı belirtilecek.
- [ ] **benchmark.py:** custom_acc yazım düzeltmeli ve düzeltmesiz olarak iki ayrı değer şeklinde raporlanacak (ablation). Düzeltmenin katkısı val setinde ölçülemiyor.
- [ ] **Sunum notları:**
  - Custom test setinde n=60 (veri v2), güven aralığı da raporlanıyor.
  - Bazı TR train cümleleri EN test cümlelerinin anlamca karşılığı (mBERT'i kayırabilir).
  - CLINC'in resmi bölmelerinde 18 kısa tekrar cümle var ("bye", "hello there").

## 9. Gereksinim denetimi bulguları (19 Kas, harfiyen kontrol)

**Cuma'ya eklenenler:**
- [ ] Quantization'dan sonra INT8 modeller `predictors.MODEL_DIRS`'e eklenecek ve `calibrate.py` ile **aynı test setinde** ölçülecek.
- [ ] `benchmark.py`:
  - "Accuracy" sütunu (tüm test seti) ve hocanın kelimeleriyle sütun adları kullanılacak.
  - Her modelin precision değeri dtype'tan okunacak.
  - Katman ve parametre sayıları config'ten yazdırılacak.
  - Yanlış tahmin edilen test cümleleri `results/errors_<model>.csv` dosyasına yazılacak (sunumdaki hata örneği için).
- [ ] Uygulama:
  - Arayüzde etiket aynen **"human agent"** olacak.
  - Sıra: ELIZA → model → iki eşik → ekip.
  - Kararı INT8 öğrenci verecek ve bu kodda açıkça görünecek.
  - **Tüm** modellerin tahmini yan yana gösterilecek.
- [ ] İsteğe bağlı (zaman kalırsa): Hugging Face Spaces'e yayınlamak. Section 1'deki *"an application that anyone can try"* ifadesini harfiyen karşılar.

**Sunum kuralları:**
- Sunumda **model kartından ya da makaleden alınan hiçbir sayı** kullanılmayacak ("104 dil" ya da "6 katman" gibi). Her sayı bizim çalıştırmamızdan gelecek.
- Özel veri **"hand-written" diye anlatılmayacak.** "AI yardımıyla yazıldı, hocanın izniyle" denecek.
- CLINC150 (Larson ve ark., 2019) ve önceden eğitilmiş modeller kaynak olarak gösterilecek.
- Gömmelerin dondurulduğu ve FFN'in `torch.optim.SGD` (mini-batch, momentum 0.9) kullandığı yazılacak.
- Sınırlılık olarak yazılacak: Test setindeki 1.000 kapsam dışı cümlenin hepsi İngilizce, Türkçe kapsam dışı mesajlar ölçülmüyor.
- Use case adı aynen yazılacak: **"Use Case 2 – Customer Support Ticket Router"**.

**Doğruluk denetimi bulguları (19 Kas gecesi):**
- **Sonuç:** Dört modelin tüm sayıları yeniden hesaplandı ve birebir tuttu. Sızıntı yok. Dinamik padding ve KD formülü doğrulandı.
- **Yöntem slaytında (aynen) yer alacak cümle:** *"We switched from one to two confidence thresholds (CLINC / appliance intents) on 19 Nov, after seeing mBERT's single-threshold test accuracy on appliance sentences (16.7%). The diagnosis and every threshold choice, including the 0.3 floor, used the validation set only. The same rule is applied to all models, and the student was trained after the rule was fixed."*
- **Tabloya dipnot:** Tek eşikli appliance acc: baseline 0.783, FFN 0.200, mBERT 0.167, student 0.283. Değerler `single_threshold_metrics` içinden alınacak.
- **Yedek slayt:** Val'deki MIN_THRESHOLD karşılaştırması (0 / 0.3 / 0.4 / 0.5).
- **Sınırlılıklar slaytı:**
  - Custom test setinin 60 cümlesinden 11'inin train'de güçlü TR↔EN karşılığı var. Bu çiftlerde doğruluk daha **düşük**. Güçlü çiftler hariç tutulunca (n=49): baseline, mBERT ve student 0.837, FFN 0.408.
  - Türkçe ya da beyaz eşyaya benzeyen kapsam dışı mesajlar ölçülmüyor.
- **Yorum kuralları:**
  - n=60'ta baseline, mBERT ve student arasındaki fark 1 cümle. "Biri daha iyi" denmeyecek.
  - Transformer'ların OOS recall'u baseline'dan düşük. Bu bir bulgu olarak açıklanacak.
  - `in_scope_acc` 4.500 CLINC cümlesine ek olarak 60 custom cümleyi de içeriyor. Dipnotta belirtilecek.
  - v1 sayıları (ör. "FFN %15") sunumda kullanılmayacak, çünkü artık dosyaları yok.
- **Kural:** Bundan sonra test sonucuna bakıp karar kuralı değiştirilmeyecek. INT8 modeller de eşiklerini aynı `calibrate.py` ile val'de alacak.
- **Teacher eğitildikten sonra kodda değişenler:** Sadece docstring ve `train_teacher.py`'den test değerlendirmesinin çıkarılması. Eğitim döngüsü, dinamik padding ve KD kodu aynı kaldı. Git olmadığı için bu kanıtlanamıyor. Bundan sonra her adım commit'lenecek.

**12 slaytlık iskelet (8–12 sınırı):**
1. Kapak: isim, öğrenci no, use case, repo linki, video linki
2. Introduction: problem, kimin için, başarı tanımı
3. Development: veri (kaynak, boyut, bölmeler, ön işleme)
4. Development: mimari diyagram (R6)
5. Development: teknik haritası (R2)
6. Development: modeller ve nasıl eğitildikleri (baseline, FFN)
7. Development: mBERT, distillation ve quantization yöntemi
8. Development: sonuç tablosu (R4)
9. Conclusion: hangi model nerede kazanıyor ya da kaybediyor
10. Conclusion: sınırlılıklar ve hata örneği
11. Conclusion: sonraki adımlar
12. Conclusion: AI kullanım slaytı (R8)

**Pazar sırası:**
1. Videoyu yükle.
2. Video linkini 1. slayta ekle ve PDF'i yeniden dışa aktar.
3. Linkleri gizli pencerede test et.
4. **Teslim formuna** repo linki, video linki ve PDF'i ekle.

**Veri izni:** Hoca derste bir soru üzerine AI ile yazılmış veriye izin verdi. Öğrenci yazılı teyidi gereksiz buldu. AI kullanım slaytında yine de belirtilecek.

## 10. Quantization ve benchmark notları (20 Kas)

**Başarı hedefleri** (Introduction slaytı). 19 Kas'ta, INT8 sonuçlarından önce konuşmada önerildi. Git geçmişi olmadığı için repoda bunun kanıtı yok, sunumda "önceden belirlenmiş" diye iddia edilmeyecek:
1. Quantize model, FP32 öğrenciye göre en fazla 1 puan Accuracy kaybetsin.
2. CPU'da mesaj başına < 50 ms ("instantly").
3. Out-of-scope recall ≥ %85.
4. Beyaz eşya (appliance) doğruluğu ≥ %80.

**Hedefleri dürüst okuma kuralları** (doğruluk denetimi, 21 Kas):
- **Hedef 1:** Sadece "Accuracy" sütununa bakıp "tuttu" denmeyecek. Şunlar birlikte verilecek:
  - Accuracy kaybı
  - In-scope kaybı (eşikli ve eşiksiz)
  - Beyaz eşya kaybı

  Eşikli in-scope farkının bir kısmı INT8'in daha yüksek CLINC eşiğinden (0.80, FP32'de 0.76) geliyor. Bu yüzden OOS recall arttı, in-scope düştü. Eşiksiz karşılaştırma `results.md` dosyasında.
- **Hedef 2:** Tutan model, karar veren INT8 model. Uygulama karşılaştırma için 6 modeli sırayla çalıştırıyor. Kullanıcının toplam beklemesi bu yüzden daha uzun (results.csv'deki gecikmelerin toplamı ~68 ms). Hedefin karar modeline ait olduğu yazılacak.
- **Hedef 3:** Güven aralığıyla birlikte yazılacak.
- **Hedef 4:** "Tutmadı" diye açıkça yazılacak, güven aralığıyla birlikte. Bu bir hata değil, bir bulgu.
- **Ablasyon:** Yazım düzeltme sadece birkaç beyaz eşya test cümlesini değiştiriyor (sayı `results.md` dosyasında). Etki birkaç cümleye dayanıyor. Ayrıca bu sadece çıkarım anında yapılan bir ablasyon, modeller düzeltilmiş metinle eğitildi.
- **"AVX2 / VNNI" açıklaması** "olası neden" diye yazılacak, test edilmedi.
- **INT8 metriklerindeki `single_threshold` değeri** FP32 öğrencinin eşiği (router_config kopyalandı). Raporlanacaksa bu belirtilecek.

Sonuç slaytında her hedef için "tuttu / tutmadı" ve güven aralığı yazılacak. Kıyasın hangi modele göre olduğu (FP32 öğrenci mi, öğretmen mi) açıkça belirtilecek.

**Quantization yöntemi:**
- ONNX Runtime dinamik INT8 (QUInt8 ağırlık, per-tensor, MatMul + Gather, MatMulConstBOnly).
- Yöntem seçimi sadece val setinde yapıldı (`src/quant_experiments.py`):
  - PyTorch dinamik INT8 bu işlemcide FP32'den yavaş çıktı.
  - ORT QInt8 doğruluğu düşürdü (AVX2, VNNI yok).
- **INT8 model her cümleyi TEK BAŞINA (batch=1) işliyor.** Dinamik quantization'da aktivasyon ölçeği batch'in tamamından hesaplanıyor. Toplu değerlendirme uygulamadaki davranıştan farklı sonuç veriyordu (doğruluk denetimi: 155/5560 tahmin farkı). Düzeltildi, INT8 eşikleri yine sadece val setinde yeniden seçildi, karar kuralı değişmedi. Bu da test sonucu görüldükten sonra yapılmış bir düzeltme, sunumda yazılacak.

**Hız kazancının kaynağı (sunumda ayrı yazılacak):**
- FP32 PyTorch → INT8 ONNX Runtime hızlanmasının bir kısmı **runtime değişikliğinden** geliyor (PyTorch FP32 → ORT FP32).
- Bir kısmı **INT8'den** geliyor (ORT FP32 → ORT INT8).
- Boyuttaki ~4× azalma tamamen INT8'den geliyor.
- Sayılar `results/quantization_experiments.csv` dosyasından alınacak. Bu CSV'deki MB ve ms tanımları `results.csv`'den farklı (tokenizer ve normalizasyon dahil değil). İki tablo karıştırılmayacak.
- mBERT FP32 bile < 50 ms hedefini tutuyor. INT8'in bu işlemcideki katkısı hedefi tutturmak değil, ~4× boyut ve ~3× hız kazancı.

**Uygulama düzeyinde metrik:** `results/app_routing.json`. ELIZA + model + eşik → **ekip** düzeyinde doğruluk. Model metriklerinden (niyet düzeyi) ayrı raporlanacak.

## 11. Teslim öncesi kontrol listesi (0/50 kapıları)

- [ ] Uygulama canlı girdi alıp sonuç veriyor (R1)
- [ ] 4 alanın hepsi var + teknik haritası slaytı (R2)
- [ ] Baseline, NN, Transformer, quantize Transformer **aynı test setinde** (R3)
- [ ] Tabloda her model için metrik + MB + ms (R4)
- [ ] Uygulama quantize modeli çalıştırıyor (R5)
- [ ] Mimari diyagram kodla birebir uyumlu (R6)
- [ ] README + tek komut + requirements.txt, büyük dosya yok (R7)
- [ ] AI kullanım slaytı (R8)
- [ ] Sunum: Giriş / Geliştirme / Sonuç, ilk slaytta isim, numara, repo ve video linki
- [ ] Video ≤ 2:00, sesli, canlı demo (en az 2 girdi + 1 zor örnek) + diyagram + tablo
- [ ] Linkler gizli pencerede açılıyor
