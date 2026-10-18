# Kaotik Kolmogorov akışında HydroGym analitik-gradyan RL

🇬🇧 [English](README.md) · 🇹🇷 Türkçe

**Akış kontrol görevi kaotikleştikçe analitik (simülatörden geri yayılımlı) politika gradyanı eklemek pekiştirmeli öğrenmeye yarar mı, zarar mı verir?**

PPO, GPPO, SAC ve gradyan destekli SAC (GSAC) algoritmalarını gerçek `hydrogym.jax` Kolmogorov akışı ortamında, düzenli (Re=45) ve kaotik (Re=300) çalışma noktalarında, beş seed üzerinde karşılaştırıyoruz. Bu depo [dynamicslab/hydrogym](https://github.com/dynamicslab/hydrogym) çatalıdır (bkz. [README_UPSTREAM.md](README_UPSTREAM.md)) ve deney kodumuz eklenmiştir.

![Dört algoritma karşılaştırması, 500 epizot](assets/demo.png)

*PPO, GPPO, SAC ve GSAC'ın iki Reynolds sayısındaki eğitim eğrileri.*

## Özet

| Problem | Yöntem | Simülasyon / donanım | Makale |
|---|---|---|---|
| Analitik gradyan eklemenin düzenli ve kaotik akış kontrolünde RL kararlılığına etkisi | PPO, GPPO, SAC, GSAC; 5 seed; kesin permütasyon testi | `hydrogym.jax` KolmogorovFlow (64x64), tek tüketici NVIDIA GPU | [paper/hydrogym-edge-of-chaos.pdf](paper/hydrogym-edge-of-chaos.pdf) |

### Sistem mimarisi

![Sistem mimarisi](assets/architecture.png)

Gerçek HydroGym Kolmogorov çözücüsü `FastKolmogorovFlow` ile sarılır (POC ölçekli zaman adımı ve ufuk); öğreniciler `real_rl.py` ve `sac_rl.py` içindedir, sonuçlar JSON olarak saklanıp şekillere dönüştürülür.

### Akış şeması (nasıl çalışır)

```mermaid
flowchart TD
  A["Algoritma ve Reynolds sayısı seç<br/>PPO / GPPO / SAC / GSAC, Re=45 veya 300"] --> B["FastKolmogorovFlow'u sıfırla<br/>(gerçek hydrogym.jax çözücüsü)"]
  B --> C["Epizot oyna, politika zorlama genliklerini seçer"]
  C --> D{"GPPO / GSAC mi?"}
  D -- "hayır" --> E["Standart PPO / SAC güncellemesi"]
  D -- "evet" --> F["jax.grad ile simülatörden geri yayılım<br/>ve RL gradyanıyla karıştır"]
  E --> G["Ödül ve gradyan normunu kaydet"]
  F --> G
  G --> H{"Sonlu olmayan değer / ıraksama?"}
  H -- "evet" --> I["Kırılma epizodunu kaydet"]
  H -- "hayır" --> J{"Epizot kaldı mı?"}
  J -- "evet" --> C
  J -- "hayır" --> K["JSON sonuçları ve logları kaydet"]
  I --> K
  K --> L["analysis/plot_paper_n5.py şekilleri üretir"]
```

## Seçilmiş sonuçlar

### Sonuç 1: Düzenli rejim (Re=45): eğitim eğrileri

GPPO, 5 seed'in 4'ünde PPO'nun en iyi ödülüne belirgin biçimde daha az epizotta ulaşıyor; HydroGym makalesinde bildirilen mekanizma yeniden üretiliyor.

![](plots/sonuc_1.png)

### Sonuç 2: Düzenli rejim (Re=45): son ödül

Dört algoritmanın beş seed üzerindeki son ödülü.

![](plots/sonuc_2.png)

### Sonuç 3: Kaotik rejim (Re=300): eğitim eğrileri

Dört algoritma da sonunda ıraksıyor.

![](plots/sonuc_3.png)

### Sonuç 4: Kaotik rejim (Re=300): kırılma epizodu

İlk ıraksama epizodunu algoritma değil seed belirliyor: kesin permütasyon testi hiçbir algoritma çiftini ayıramıyor (hepsi p >= 0,19).

![](plots/sonuc_4.png)

### Sonuç 5: Kırılmanın mekanizması

Kırılma ani ve öncül işaretsiz; tetikleme anındaki replay buffer boyutu, tetiklemeden sonra kısa bir hayatta kalma penceresi açılıp açılmayacağını belirliyor.

![](plots/sonuc_5.png)

### Sonuç 6: Analitik gradyan normları

Kaotik rejimdeki kırılmaya eşlik eden gradyan normu davranışı.

![](plots/sonuc_6.png)

*Tüm şekiller `plots/` klasöründe.*

## Kurulum

Gereksinimler: Linux, Python 3.10 veya 3.11, sürücüsü >= 550 olan NVIDIA GPU (sistem CUDA gerekmez), yaklaşık 8 GB boş disk.

```bash
git clone https://github.com/GulogluHaktan/hydrogym-work.git
cd hydrogym-work
bash scripts/setup_env.sh                # GPU JAX ile ../venv oluşturur (yaklaşık 6 GB)
source ../venv/bin/activate
python -c "import jax, hydrogym.jax; print(jax.devices())"
```

## Kullanım

Kısa tek konfigürasyon çalıştırma (Re=45'te PPO, 25 epizot):

```bash
python experiments/run_real_experiment.py --episodes 25 --only ppo_low --out results/results_demo.json
```

Beklenen çıktı: epizot başına düzenli log satırları (ödül, gradyan normu) ve bir `results/results_demo.json` dosyası.

## Sonuçları yeniden üret

| Şekil | Komut |
|---|---|
| Sonuç 1-4, mekanizma ve gradyan normu şekilleri (`plots/sonuc_1..6.png`) | `bash scripts/make_readme_plots.sh` (`analysis/plot_paper_n5.py` çalıştırır, `paper/elsarticle/figures/fig_*.pdf` yazar) |
| Seed 1-4 verisi (seed başına yaklaşık 8,5 saat) | `bash scripts/run_seed_sweep.sh <SEED>` |
| Seed 0 verisi | `bash scripts/legacy/run_ppo_gppo_500_sequential.sh` ardından `bash scripts/legacy/run_sac_gsac_500_sequential.sh` |

Tüm seed'lerin sonuç JSON dosyaları depoda olduğundan şekiller yeniden eğitim yapmadan üretilebilir.

## Proje yapısı

```
hydrogym_real/
├── experiments/           # kendi kodumuz
│   ├── fast_env.py        #   FastKolmogorovFlow: gerçek ortam için POC ölçekli parametreler
│   ├── real_rl.py         #   PPO / GPPO
│   ├── sac_rl.py          #   SAC / GSAC
│   ├── safeguards.py      #   gradyan kırpma, sonlu-olmayan geri dönüş, ısınma, uyarlanır karışım
│   ├── run_real_experiment.py  # PPO/GPPO sürücüsü
│   ├── run_sac_experiment.py   # SAC/GSAC sürücüsü
│   └── real_grad_probe.py #   gradyan duyarlılık sondası
├── analysis/              # grafik ve istatistik (plot_paper_n5.py, sig_test_n5.py, ...)
├── scripts/               # setup_env.sh, run_seed_sweep.sh, run_ext_all.sh, make_readme_plots.sh
├── results/               # konfigürasyon başına sonuçlar, results_500_*.json (500 epizot)
├── logs/                  # eğitim logları (git-ignore)
├── plots/                 # sonuç grafikleri sonuc_1..6.png
├── paper/                 # makale: PDF, PAPER_DRAFT_EN.md, jai/ (JAI), elsarticle/ (Elsevier)
├── reports/               # internal reports (Türkçe) ve PDF üreticileri
├── assets/                # README görselleri
├── hydrogym/              # upstream HydroGym kütüphanesi (fork), see README_UPSTREAM.md
├── docs/ examples/ test/ third_party/   # upstream
├── requirements.txt
└── README.md
```

## Atıf, lisans ve iletişim

```bibtex
@misc{ozkurt2026gradient,
  title  = {Gradient at the Edge of Chaos: Why Analytic-Gradient Reinforcement
            Learning Has No Seed-Robust Advantage on Chaotic Kolmogorov Flow},
  author = {{\"O}zkurt, Cem and Onay, Muhammed Emin and Tun{\c{c}}, Ey{\"u}p Altu{\u{g}}
            and K{\"u}{\c{c}}{\"u}kler, Ahmet Kutay and G{\"u}lo{\u{g}}lu, Haktan Yusuf},
  year   = {2026},
  note   = {Manuscript in preparation. Code archive DOI: TODO (Zenodo)}
}
```

Lisans: MIT ([LICENSE](LICENSE), upstream HydroGym'den devralındı).
İletişim: Cem Özkurt (sorumlu yazar), cemozkurt@sakarya.edu.tr.
