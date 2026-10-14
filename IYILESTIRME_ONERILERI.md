# HydroGym Makalesi — Mevcut Durum ve Önerilen Ek Çalışmalar

*Hazırlık: 2026-08-31. Karar için üst yönetime sunulmak üzere.*

---

## 1. Şu an elimizde ne var (n=5 tamamlandı)

**Kurulum.** 2×2 tasarım — {on-policy PPO, off-policy SAC} × {analitik gradyan var / yok} —
gerçek HydroGym `hydrogym.jax` Kolmogorov akış ortamında (64×64, türevlenebilir sözde-spektral
çözücü). İki rejim: Re=45 (düzenli), Re=300 (kaotik; çözücü kararlılık tavanı ~350–380
olduğu için 450 yerine 300). Config başına 500 epizot, **5 bağımsız seed**.

**GSAC:** SAC aktör gradyanına, simülatör içinden geri-yayılan kısa-ufuklu (H=6) analitik
gradyanın 50/50 karıştırılması. Bir PDE-kontrol benchmark'ında ilk kez uygulanıyor.

### Sonuçlar

**Re=45 (düzenli):** GPPO, PPO'nun en iyi ödülüne 5 seed'in 4'ünde belirgin şekilde daha az
epizotta ulaşıyor (~%63–84; seed=0'da %63, Nature makalesinin %65'ine yakın). Off-policy aile
(SAC/GSAC) her seed'de daha iyi final ödül. → Nature makalesinin çekirdek iddiası düzenli
rejimde **doğrulanıyor**.

**Re=300 (kaotik) — ana bulgu:** Dört algoritma da eninde sonunda ıraksıyor (NaN). İlk-NaN
epizodu:

| Algoritma | s0 | s1 | s2 | s3 | s4 | medyan | ortalama |
|---|---|---|---|---|---|---|---|
| PPO  | 12 | 35 | 117 | 14 | 11 | 14 | 37.8 |
| SAC  | 36 | 6 | 2 | 40 | 131 | 36 | 43.0 |
| GPPO | 83 | 40 | 32 | 6 | 16 | 32 | 35.4 |
| GSAC | 141 | 63 | 2 | 45 | 122 | **63** | **74.6** |

**İstatistik testi (kesin işaret-çevirme permütasyon + bootstrap + omnibus):**
hiçbir karşılaştırma anlamlı değil.
- GPPO − PPO: ortalama **−2.4 epizot**, p = 0.81 → analitik gradyan on-policy kırılmayı
  **geciktirmiyor**.
- GSAC − SAC p = 0.38 · GSAC − GPPO p = 0.19 · GSAC − PPO p = 0.38.
- Kruskal–Wallis p = 0.57 · Friedman p = 0.77.

**→ Bulgu: kaotik rejimde eğitim kararlılığının algoritmaya göre seed-sağlam bir sıralaması
yok.** Kırılma epizodu seed-baskın bir nicelik (algoritma-içi seed-arası std 30–57 epizot,
algoritma-arası ortalama farkı yalnızca 35–75 aralığında). GSAC'ın en yüksek medyanı var ve
5 seed'in 3'ünde en sağlam — ama bu anlamlı değil, "aday etki" olarak raporlanıyor.

**Ayakta kalan mekanizma bulguları:**
- **§6(a):** Kırılma ani ve öncüsüz — kırılmadan önceki 40 epizotta ödül/gradyan-normu düz.
  Politikanın kademeli olarak çözücüyü zorlaması değil; tek bir epizodun rastgele aksiyon
  dizisinin akışı kararlılık sınırının ötesine itmesi. (4 seed'de doğrulandı.)
- **§6(b):** Tetik sonrası kısa "hayatta kalma penceresi" yalnızca replay buffer tetik anında
  yeterince büyükse (~%1 zehirli oran) mümkün. 20 Re=300 koşusundan yalnızca 1'inde görüldü
  (seed=0 GSAC, 2 epizot). Düşündürücü, kanıtlanmış değil.
- **§6(c):** Nadir bir modda (5 seed'den 1'i — seed=2) akış öğrenme başlamadan ~2 epizotta
  ıraksıyor; çıplak SAC ve GSAC birebir aynı davranıyor → çevresel, algoritma kaynaklı değil.

### Makalenin mevcut değeri

Titiz bir **replikasyon + sınır-koşulu + yöntem-varyantı + mekanizma** makalesi. SOTA iddiası
yok. Değeri: türevlenebilir-simülasyon RL topluluğunun aktif olarak tartıştığı bir soruda
(kaos analitik gradyanı bozar mı?) gerçek bir benchmark üzerinde nicel, dürüst bir cevap.
**Gerçekçi hedef:** MLST veya benzeri Q1-alan dergisi — aşağıdaki ek çalışmalarla.

---

## 2. Önerilen ek çalışmalar

Her paket bağımsız değerlendirilebilir. GPU: tek RTX 5060 (dizüstü), eşzamanlı JAX
çalıştıramıyor, tüm süreler seri.

| # | Paket | GPU-saat | Takvim | Ne kazandırır | Risk |
|---|---|---|---|---|---|
| A | Reynolds taraması {100,150,200,250} | ~35–40 | 1 hafta | **Ana yeniden-çerçeveleme.** "Re=300'de null" → "sıralama Re≈X üstünde kayboluyor" | Düşük |
| B | Güçlü PPO + resmi HydroGym GPPO mutabakatı | ~10 + geliştirme | 4–6 gün | Geçerlilik sigortası: "PPO'nuz zayıf" itirazını kapatır | Orta (resmi kod tavşan deliği olabilir) |
| C | GSAC-taban korumaları (grad-clip / non-finite fallback / warmup / adaptif ağırlık) | ~20 | 1–1.5 hafta | Pozitif yöntem katkısı: GSAC kırılma varyansını düşürme | Orta-yüksek (ep-2 modu düzelmeyebilir) |
| D | 128×128 doğrulama koşusu | ~15 | 1 hafta | "Oyuncak grid" itirazını kapatır | Düşük |
| E | Re=45'te H ∈ {3,6,12} ve karışım oranı ∈ {0.25,0.5,0.75} ablasyonu | ~40 | 1 hafta | Mekanizma sağlamlaştırma; hiperparametre duyarlılığı | Düşük |
| F | Seed sayısını artırma (n=5 → n=10+) | ~50 (5 seed) | 1.5 hafta | GSAC'ın anlamsız medyan eğilimini test eder | Düşük ama getirisi belirsiz |

### Paket detayları

**A — Reynolds taraması. [EN YÜKSEK ÖNCELİK]**
Re ∈ {100, 150, 200, 250} × 4 algoritma (yalnız _high configleri) × 3 seed.
- *Amaç:* Analitik-gradyan RL'in seed-sağlamlığını kaybettiği eşiği bulmak.
- *Neden ilk:* (i) En büyük hakem itirazını ("tek nokta / Re=300 gerçekten kaotik değil")
  kapatır. (ii) Paket C'nin ön koşulu — GSAC korumasının hangi Re'de çalışması gerektiğini
  bu belirler. (iii) Makaleyi "null sonuç"tan "işte geçiş noktası" pozitif bulgusuna çevirir.
- *Çıktı:* İlk-NaN epizodu vs Re grafiği, algoritma başına; Boffetta–Musacchio öngörülebilirlik
  ufkuyla çerçevelenmiş.
- *Risk:* Düşük — sonuç ne çıkarsa çıksın yayınlanabilir.

**B — Güçlü PPO baseline + resmi GPPO mutabakatı.**
Mevcut PPO epizot başına tek tam-batch gradyan adımı atıyor (epoch/minibatch yok). Bu "PPO
zayıf" itirazına açık ve tüm null bulgusunu tehdit eder.
- *Ne yapılacak:* PPO'ya epoch + minibatch ekle, `ppo_high`/`gppo_high` × 5 seed yeniden koş.
  Ayrı olarak resmi HydroGym GPPO implementasyonunu çalıştırıp bizim GPPO ile karşılaştır.
- *Risk:* Resmi kodu çalıştırmak (Firedrake bağımlılıkları) birkaç gün sürebilir.

**C — GSAC-taban korumaları.**
İki modu ayır:
- *(a) Çevresel ep-2 çöküşü (§6c):* muhtemelen algoritma tarafından düzeltilemez. Karakterize
  et, geç.
- *(b) Orta-eğitim kırılma epizodunun seed-varyansı:* buna saldır. Analitik-gradyan-normu
  tetikli fallback, kısa warmup, adaptif karışım ağırlığı (GI-PPO α-policy tarzı).
- *Hedef iddia:* "norm-kapılı fallback GSAC'ın kırılma-epizodu varyansını düşürüyor / medyanı
  yükseltiyor, Re ∈ [X, 300]." — "GSAC artık NaN vermiyor" **değil**.
- *Değerlendirme:* `gsac_high` × 5 seed × 4 varyant.
- *Risk:* Orta-yüksek. En kötü senaryo: korumalar varyansı anlamlı düşürmez → ikinci bir
  negatif sonuç (yine de raporlanabilir ama zayıf).

**D — 128×128 doğrulama.**
1–2 config (ör. `gppo_high`, `gsac_high`) × 1–2 seed, 128×128 gridde. DOF 4×, duvar süresi
~3–4×. "Bulgular oyuncak gridde" eleştirisini kapatır; nicel eşleşme beklenmez, niteliksel
tekrar yeterli.

**E — Re=45 ablasyonları.**
H ve karışım oranının GPPO/GSAC örneklem verimliliğine etkisi. Mekanizmayı sağlamlaştırır,
"50/50 ve H=6 keyfi mi" sorusunu yanıtlar. Düşük risk, orta getiri.

**F — Daha fazla seed.**
n=5 → n=10+. GSAC'ın anlamsız medyan eğilimini (p ≈ 0.19–0.38) test etmenin tek yolu. Ama
seed-baskınlık göz önüne alınırca kesin bir test çok sayıda seed isteyebilir; getiri belirsiz.

---

## 3. Öneri: iki senaryo

### Minimum (makaleyi savunulabilir Q1'e taşır) — ~3 hafta
**A + B.** Reynolds taraması yeniden-çerçeveleme sağlar; güçlü PPO geçerliliği güvenceye alır.
Makale: "Analitik-gradyan RL Kolmogorov akışında Re≈X üstünde seed-sağlam avantajını
kaybediyor; mekanizma §6'da." — solid negatif/sınır makalesi.

### Tam (pozitif yöntem katkısı da ekler) — ~6 hafta
**A + B + C + D.** C başarılı olursa makale: "...ve GSAC'ın kararlılık varyansı norm-kapılı
fallback ile azaltılabiliyor." — hem sınır bulgusu hem yöntem katkısı. E ve F opsiyonel,
revizyon aşamasında hakem isterse.

### Karar noktası
C paketi riskli (ep-2 modu çevresel, düzelmeyebilir). Öneri: **önce A + B yapılsın**, A'nın
sonucu görülünce C'ye girilip girilmeyeceğine karar verilsin — çünkü A, C'nin hangi Re'de
anlamlı olduğunu söyler.

---

## 4. Bağımsız repro paketi (her senaryoda gerekli, ~2 gün)
Deney kodunu upstream HydroGym klonundan ayrı bir repoya çıkar; `requirements-lock.txt`,
`setup_env.sh`, `run_seed_sweep.sh` zaten hazır. ML-reproducibility-checklist atıfı ekle.
