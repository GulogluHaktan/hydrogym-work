# HydroGym — Ek Çalışmalar Paketi (A + B + C)

Bu paket, `IYILESTIRME_ONERILERI.md` içindeki üç ek çalışmayı **atölye
bilgisayarında tek komutla** çalıştırmak için hazırlandı.

| Paket | Ne yapar | Nasıl |
|---|---|---|
| **A** | Reynolds taraması `{100,150,200,250}` × 4 algoritma × 3 seed | mevcut kod, sadece `--re` parametresi |
| **B** | Güçlü PPO baseline — epizot başına tek adım yerine gerçek PPO iç-döngüsü (K epoch × M minibatch) | `real_rl.py` içine eklendi, varsayılan davranış değişmedi |
| **C** | GSAC-taban korumaları — analitik-gradyan kırpma / non-finite fallback / warmup / adaptif karışım | `sac_rl.py` + yeni `safeguards.py`, varsayılan davranış değişmedi |

Kod, varsayılan argümanlarla (bayrak verilmezse) **eski seed=0..4 taramasıyla
bit-aynı** çalışır — yeni davranışların hepsi opsiyonel bayrakların arkasında.

---

## Kurulum + çalıştırma (tek komut)

```bash
git clone https://github.com/GulogluHaktan/hydrogym-work.git
cd hydrogym-work
nohup bash run_ext_all.sh > log_ext_all.log 2>&1 &
tail -f log_ext_all.log
```

`run_ext_all.sh`:
1. `venv/` yoksa `setup_ext_env.sh` ile kurar (~10–15 dk, ağ gerekir, ~6 GB).
2. A → B → C paketlerini **sırayla** koşar (tek GPU, eşzamanlı JAX yok).
3. Her config kendi JSON'una yazar; **dosya varsa atlar** → kesilirse aynı
   komutla kaldığı yerden devam eder.
4. Bitince `analyze_ext.py` çalışır → **`EXT_SONUCLAR.md`** + (matplotlib
   varsa) `results_ext/fig_ext_re_sweep.pdf`.

### Gereksinimler
- Python 3.10 veya 3.11
- NVIDIA GPU + sürücü ≥ 550 (sistem CUDA gerekmez; RTX 5060 8 GB yeterli)
- ~8 GB boş disk

### Süre (RTX 5060 dizüstü referansı, 500 epizot)
| Paket | Koşu sayısı | Tahmini süre |
|---|---|---|
| A | 4 Re × 4 algo × 3 seed = 48 | ~35–40 saat |
| B | 2 config × 5 seed = 10 | ~8–12 saat |
| C | 5 varyant × 5 seed = 25 | ~25–30 saat |
| **Toplam** | **83** | **~3 gün kesintisiz** |

---

## Kapsamı daraltma (ortam değişkenleri)

```bash
# Sadece Reynolds taraması, 300 epizot, 2 seed:
EPISODES=300 PACKAGES="A" SEEDS_A="0 1" bash run_ext_all.sh

# Sadece paket C:
PACKAGES="C" bash run_ext_all.sh

# Farklı Re listesi:
RE_LIST="120 180 240" bash run_ext_all.sh
```

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `EPISODES` | `500` | config başına epizot |
| `PACKAGES` | `A B C` | koşulacak paketler |
| `RE_LIST` | `100 150 200 250` | paket A Reynolds değerleri |
| `SEEDS_A` | `0 1 2` | paket A seed'leri |
| `SEEDS_B` | `0 1 2 3 4` | paket B seed'leri |
| `SEEDS_C` | `0 1 2 3 4` | paket C seed'leri |
| `PPO_EPOCHS` | `4` | paket B: PPO iç-döngü epoch sayısı (1 = eski davranış; ≥10 Re=300'de çok agresif, hızlı ıraksıyor) |
| `PPO_MINIB` | `1` | paket B: minibatch sayısı (batch = 15) |

---

## Tek tek çalıştırma (istersen)

```bash
source venv/bin/activate
export PYTHONPATH="$PWD"

# Paket A — bir nokta
python run_real_experiment.py --episodes 500 --seed 0 --only gppo_high --re 150 \
    --out results_ext/A_gppo_re150_seed0.json

# Paket B — güçlü PPO
python run_real_experiment.py --episodes 500 --seed 0 --only ppo_high \
    --ppo-epochs 4 --ppo-minibatches 1 --out results_ext/B_ppo_strong_seed0.json

# Paket C — bir koruma varyantı
python run_sac_experiment.py --episodes 500 --seed 0 --only gsac_high \
    --grad-clip 15 --nonfinite-fallback --analytic-warmup 25 --adaptive-mix \
    --out results_ext/C_gsac_combo_seed0.json

# Analizi elle yeniden üret
python analyze_ext.py --dir results_ext --out EXT_SONUCLAR.md
```

### Yeni bayraklar
| Bayrak | Script | Paket | Anlam |
|---|---|---|---|
| `--re FLOAT` | her ikisi | A | seçili config'in Re'sini değiştir |
| `--ppo-epochs N` | real | B | PPO iç-döngü geçiş sayısı (1 = eski davranış) |
| `--ppo-minibatches N` | real | B | minibatch sayısı (1 = eski davranış) |
| `--grad-clip X` | her ikisi | C | analitik gradyanı global-norm X'e kırp |
| `--nonfinite-fallback` | her ikisi | C | non-finite analitik gradyanda o epizot saf model-free adım |
| `--analytic-warmup N` | her ikisi | C | ilk N epizot analitik enjeksiyon yok |
| `--adaptive-mix` | her ikisi | C | norm `--adaptive-target`'ı aşınca analitik ağırlığı küçült |

---

## Çıktı dosyaları

```
hydrogym_ext/
├── EXT_SONUCLAR.md              <- okunabilir özet tablolar (Türkçe)
├── results_ext/
│   ├── A_{algo}_re{RE}_seed{S}.json
│   ├── B_{ppo,gppo}_strong_seed{S}.json
│   ├── C_gsac_{clip,fallback,warmup,adaptive,combo}_seed{S}.json
│   ├── *.log                    <- her koşunun stdout'u
│   └── fig_ext_re_sweep.pdf     <- paket A grafiği
└── log_ext_all.log             <- ana sürücü logu
```

Her JSON'da yeni alanlar: `ext_config` (kullanılan bayraklar),
`mix_weights` (epizot başına analitik ağırlık), `safeguard_events`
(hangi epizotta hangi koruma tetiklendi).

---

## Notlar / sınırlar

- **B paketi dürüst sınırı:** "güçlü PPO" burada = K-epoch × M-minibatch iç
  döngü + (zaten açık olan) avantaj normalizasyonu. Ayrı olarak resmi
  HydroGym GPPO implementasyonuyla mutabakat (memo'daki B'nin ikinci yarısı)
  bu pakete **dahil değil** — Firedrake bağımlılıkları ayrı bir iş.
- **C paketi hedefi:** kırılma-epizodu **varyansını düşürmek**, "GSAC artık
  NaN vermiyor" demek değil. §6c'deki çevresel ep-2 çöküşünün korumalarla
  düzelmesi beklenmiyor; `analyze_ext.py` bunu varyant başına std ile raporlar.
- Taban çizgisi karşılaştırması `analyze_ext.py` içine gömülü n=5 sayılarıyla
  yapılır (Re=300, seed 0..4 — `IYILESTIRME_ONERILERI.md` tablosu).
- Tek GPU'da her şey seri; `run_ext_all.sh` idempotent, güvenle tekrar çalıştırılır.
