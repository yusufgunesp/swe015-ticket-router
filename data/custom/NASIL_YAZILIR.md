# Kendi veri setimiz: `appliance_tr_en.csv`

**300 satır:** 5 niyet × (40 Türkçe + 20 İngilizce).

**Nasıl üretildi:** Cümleler AI yardımıyla (Claude) yazıldı ve elle gözden geçirildi. Hoca derste bir soru üzerine izin verdi. Sunumda bu veri **"hand-written" olarak tanıtılmaz**, AI kullanım slaytında ayrıca belirtilir.

**Sürümler:**
- **v1, 18 Kas:** 100 cümle (5 × 10 TR + 10 EN).
- **v2, 19 Kas:** Her niyete 30 TR + 10 EN cümle eklendi, toplam 300. Sebep: v1'de eğitim setinde sadece 30 Türkçe cümle vardı. Ödev 50–100 örneği *öneriyor* ("You may add"). 100'ün aşılmasının gerekçesi Türkçe kapsamı.

## Dosya biçimi

```csv
text,intent,lang
"bulaşık makinem su boşaltmıyor, model DW-6534",dishwasher_issue,tr
my washing machine makes a loud noise when spinning,washing_machine_issue,en
```

## 5 niyet (intent)

| intent | Ne hakkında? | Yönlendirilen ekip |
|---|---|---|
| `dishwasher_issue` | Bulaşık makinesi arızası (su boşaltmıyor, yıkamıyor, hata kodu…) | Teknik Servis – Bulaşık Makinesi |
| `washing_machine_issue` | Çamaşır makinesi arızası (sıkmıyor, ses yapıyor, kapak açılmıyor…) | Teknik Servis – Çamaşır Makinesi |
| `refrigerator_issue` | Buzdolabı arızası (soğutmuyor, buzlanma, su akıtıyor…) | Teknik Servis – Buzdolabı |
| `warranty_inquiry` | Garanti süresi, garanti kapsamı, belge | Garanti & Satış Sonrası |
| `technician_appointment` | Servis randevusu alma, değiştirme, iptal | Randevu Planlama |

## Çeşitlilik (modelleri zorlamak için)

- Bazı cümlelerde **model kodu** var: `DW-6534`, `WM-1200`, `RF-450X` (regex ile çıkarılıyor).
- Bazılarında ürün adı **yanlış yazılmış**: `bulasik makinsi`, `dishwaser`, `refridgerator` (edit distance ile düzeltiliyor).
- **Türkçe karakter kullanılmadan** yazılmış cümleler var: `bulasik makinem calismiyor`.
- **Çeviri çiftleri (doğruluk denetimi, 19 Kas):** Cümlelerin bir kısmı birbirinin TR↔EN karşılığı ya da anlamca çok yakını. Örnek: "teknisyen geldi ama parça yoktu…" ↔ "the technician came but didn't have the part…". Bu hem v1–v2 arasında hem v2'nin kendi içinde var. Test setindeki 60 cümlenin 11'inin train'de güçlü, 5'inin orta düzeyde karşılığı bulunuyor. Bölme değiştirilmedi, çünkü bu da test'e bakarak yapılmış bir değişiklik olurdu. Sınırlılık olarak raporlanıyor.
