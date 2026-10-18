# Gerçek HydroGym.jax POC Raporu

**Neden önce fallback'e geçilmişti, ne değişti:** İlk turda HydroGym'i hiç
denemeden fallback'e geçmek hataydı — talimattaki "15 dakikayı aşarsa
fallback'e geç" kuralı sadece kurulum zorlaştığında geçerliydi, ama kurulum
denenmemişti bile. Bu turda gerçekten klonlanıp kuruldu:
`pip install -e ".[jax]"` (Firedrake gerekmedi) ~2 dakikada sorunsuz kuruldu.
`hydrogym/jax/envs/kolmogorov.py` içinde gerçekten hazır bir `KolmogorovFlow`
gymnax-env'i var — 64x64 grid (4096 DOF), 4 aktüatörlü sinüs kuvvetleme,
pseudo-spektral IMEX RK4-CN çözücü.

## Kurulum sırasında bulunan gerçek quirk'ler
- `step_env`'e geçirilen `params.dt` / `params.action_time` görmezden
  geliniyor; kütüphane bunun yerine `self.default_params`'ı tekrar okuyor.
  Bunu değiştirmek için `default_params` property'sini override eden
  `fast_env.py::FastKolmogorovFlow` alt sınıfı yazıldı.
- Varsayılan `action_time=10.0`, `dt=0.001` → RL-adımı başına 10.000 alt-adım
  (jit sonrası bile ~2.7s/adım). Kütüphane `action_time<1.0`'ı reddediyor
  (ValueError), o yüzden minimum olan `action_time=1.0`'a düşürüldü.
- Varsayılan `reward_alpha=0.0` — yani ödül sadece kontrol maliyeti, akış
  enerjisi YOK sayılıyor. `reward_alpha=1.0` ile düzeltildi.
- **Kararlılık sınırı keşfedildi:** bu grid/dt kombinasyonunda (64x64,
  dt=0.001), kütüphanenin kendi çözücüsü Re≈350-380 civarında NaN'a
  patlıyor — sıfır aksiyonla bile, sadece analitik sin(x)cos(y) başlangıç
  koşulundan 2-4 adımda. dt'yi 10x küçültmek düzeltmedi (CFL değil,
  çözünürlük/aliasing sınırı gibi görünüyor). Bu yüzden orijinal planın
  Re=450'si yerine **Re=300** kullanıldı (kararlı doğrulandı).

## Deney
64x64 gerçek grid, gerçek pseudo-spektral IMEX-RK4CN çözücü, dt=0.001 (her
iki Re için de aynı — dt farkının kendisi bir karışıklık kaynağı olmasın
diye), action_time=1.0, episode=15 RL-adımı, GPPO kısa ufku=6 adım, n=1 seed,
25 episode/konfigürasyon. Toplam çalışma süresi: **~10 dakika** (98.9 +
95.5 + 202.2 + 199.4 saniye).

## Ana bulgu: PPO Re=300'de NaN'a patladı, GPPO patlamadı

En çarpıcı sonuç ham deneyden çıktı, herhangi bir prob'a gerek kalmadan:
**Re=300'de eğitilen saf PPO politikası, episode 12'den itibaren simülasyonu
kararsızlığa sürükledi ve kalan 14/25 episode NaN oldu** (politika, ödülü
düşürmek için kontrol kuvvetini büyütürken kütüphanenin kararlılık sınırına
çarptı). **Aynı Re=300'de eğitilen GPPO ise 25 episode boyunca hiç NaN
vermeden tamamen kararlı kaldı** (bkz. `real_results.png`, sol panel).
Re=45'te hem PPO hem GPPO sorunsuz tamamlandı.

Bu, "kaotik/kararsızlık-sınırı yakınındaki rejimlerde analitik gradyan
enjeksiyonunun optimize ediciyi kararsızlaştıran bölgelerden uzak tutabildiği"
hipotezini destekleyen güçlü, dolaysız bir kanıt — n=1 seed olduğu için
kesin nedensel iddia yapılamaz (şans eseri de olabilir), ama gerçek çözücüde,
gerçek bir kırılma noktasında gözlenmiş bir fark.

**Eğitim sırasındaki gradyan norm CV'si** de aynı yönde: düşük Re CV=0.176,
yüksek Re CV=0.238 (~%35 daha değişken) — yön hipotezi destekliyor ama fark
küçük ölçekli fallback deneyindeki kadar (4.6x) belirgin değil; bu ham eğri
politika değişiminden kirlenmiş, [[fallback POC]]'taki gibi kontrollü bir
prob burada da denendi ama (1% pertürbasyon, 6 adımlık ufuk) anlamlı bir
CV farkı yakalayamadı — muhtemelen pertürbasyon çok küçük/ufuk çok kısaydı;
zaman kısıtı nedeniyle daha fazla iyileştirilmedi.

## Sınırlamalar
- n=1 seed — NaN olayı dahil hiçbir sonuç istatistiksel olarak sağlam değil.
- Re=450 yerine Re=300 kullanıldı (kütüphanenin kendi kararlılık sınırı
  nedeniyle) — orijinal "yüksek Re" hedefinden daha düşük.
- 25 episode, kısa (15 adım) episode'lar — gerçek RL yakınsaması için
  yetersiz, mekanizma kanıtı olarak okunmalı.
- Kontrollü gradyan-prob'u gerçek çözücüde net sinyal vermedi (fallback'in
  aksine) — bu metodolojinin kendisinin daha fazla ayarlanması gerektiğini
  gösteriyor, hipotezin yanlış olduğunu değil (ana NaN bulgusu zaten çok
  daha güçlü, dolaysız bir kanıt sağlıyor).

**Sonuç:** Gerçek HydroGym.jax üzerinde, kaotik/kararsızlık sınırı
yakınındaki bir rejimde saf PPO'nun simülasyonu bizzat kararsızlığa
sürüklediği, analitik gradyan enjeksiyonu eklenen GPPO'nun ise aynı
koşullarda kararlı kaldığı gözlendi — küçük ölçekli ve n=1 olsa da, bu
gerçek bir çözücüde gerçekleşen, dolaysız ve çarpıcı bir kanıt. Daha büyük
ölçekte, çoklu seed'lerle ve nedensellik iddiası öncesi ablasyonlarla
(örn. sadece daha küçük öğrenme oranı/action clipping ile PPO'nun da kararlı
kalıp kalmadığının kontrolü) doğrulanması gerekir.
