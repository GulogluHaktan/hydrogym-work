# Gerçek HydroGym.jax entegrasyonu notları

- Repo: github.com/dynamicslab/hydrogym (poetry-core backend, `pip install -e ".[jax]"` ile
  Firedrake gerekmeden kuruldu, ~2 dakika sürdü).
- `hydrogym/jax/envs/kolmogorov.py` gerçekten var: gymnax tarzı KolmogorovFlow env,
  64x64 grid (4096 DOF), action_dim=4 (4 sinüs modu forcing genliği), obs_dim=64
  (8x8 alt-örneklenmiş hız büyüklüğü).
- Kritik quirk: `step_env`'e geçirilen `params.dt` / `params.action_time` GÖRMEZDEN
  GELİNİYOR; `_rollout` bunun yerine `self.default_params`'ı tekrar okuyor. Bu yüzden
  dt/action_time'ı değiştirmek için `default_params` property'sini override eden bir
  alt sınıf (`fast_env.py::FastKolmogorovFlow`) yazmak gerekti.
- Kütüphane varsayılanı: dt=0.001, action_time=10.0 (episode başına 10.000 alt-adım) —
  bir RL-adımı jit sonrası bile ~2.7s. Tek gecelik POC için action_time'ı kütüphanenin
  izin verdiği minimum olan 1.0'a düşürdük (action_time<1 ValueError fırlatıyor).
- `reward_alpha` varsayılanı 0.0 — yani ödül sadece kontrol maliyeti, akış enerjisi
  YOK sayılıyor. `reward_alpha=1.0` ile düzelttik.
- Kararlılık sınırı keşfedildi: bu grid/dt kombinasyonunda (64x64, dt=0.001), kütüphanenin
  KENDİ solver'ı Re~350-380 civarında NaN'a patlıyor (Re=350 kararlı, Re=380+ 2-4 adımda
  NaN) — sıfır aksiyonla bile, sadece analitik sin(x)cos(y) başlangıç koşulundan. dt'yi
  10x küçültmek instabiliteyi düzeltmedi (bu bir CFL sorunu değil, çözünürlük/aliasing
  sınırı gibi görünüyor). Bu yüzden "yüksek Re" için 450 yerine 300 kullanıldı (300 hem
  15 adımlık tam episode hem de rastgele politika aksiyonlarıyla test edildi, kararlı).

## Ek çalışmalar paketi (A + B + C) — 2026-09-02

`IYILESTIRME_ONERILERI.md` paketleri A/B/C için `real_rl.py` / `sac_rl.py` /
sürücü scriptlerine **eklemeli** (varsayılanı değiştirmeyen) parametreler eklendi.
Bayrak verilmezse eski seed=0..4 taraması **bit-aynı** (PPO tam-aynı; SAC ~1e-5
düzeyinde — bu GPU/XLA reduction sırası kaynaklı, koşudan koşuya da var, yeni kod
kaynaklı değil).

- **Yeni dosya `safeguards.py`**: `SafeguardConfig` + `process_analytic()`. Paket C'nin
  dört korumasını (grad-clip / non-finite fallback / analytic-warmup / adaptive-mix)
  tek yerde tutar; `real_rl.py` (GPPO) ve `sac_rl.py` (GSAC) ikisi de import eder
  (CLAUDE.md "üçünde tutarlı" kuralı — `rl.py` fallback track'ine dokunulmadı).
- **`real_rl.train()` yeni argümanlar**: `ppo_epochs`, `ppo_minibatches` (paket B —
  epizot başına tek tam-batch adım yerine gerçek PPO iç-döngüsü; `strong` yolu
  ayrı `if` bloğu, varsayılan `1,1` orijinal yolu aynen çalıştırır), `sg` (paket C).
- **`sac_rl.train()` yeni argüman**: `sg` (paket C, GSAC). Analitik terim düşse bile
  ikinci aktör adımı (saf SAC) atılır → epizot başına güncelleme sayısı sabit.
- **Sürücü bayrakları**: `--re` (paket A, config Re override), `--ppo-epochs`,
  `--ppo-minibatches`, `--grad-clip`, `--nonfinite-fallback`, `--analytic-warmup`,
  `--adaptive-mix`, `--adaptive-target`.
- **JSON'a eklenen alanlar**: `ext_config`, `mix_weights`, `safeguard_events`.
- **`run_ext_all.sh`**: tek komut sürücü (A→B→C seri, resume'lu). `analyze_ext.py`
  → `EXT_SONUCLAR.md` + `results_ext/fig_ext_re_sweep.pdf`. Taşınabilir zip:
  `setup_ext_env.sh` venv'i paket içine kurar, `PYTHONPATH` ile gömülü `hydrogym`
  (firedrake'siz) — `pip install -e .` / poetry-core gerekmez.
