# Kaos Sınırında Gradyan: HydroGym'de Analitik Gradyan Yöntemlerinin Kırılma Noktası

*(Alt başlık: HydroGym Kolmogorov Akışında PPO, GPPO, SAC ve GSAC'ın Kırılma-Noktası Karşılaştırması)*

**Yazar:** Haktan Güloğlu
**Kurum:** Sakarya Üniversitesi, Milli Teknoloji Atölyesi, Yapay Zeka Merkezi
**İletişim:** haktanguloglu24@gmail.com

**Durum:** Genişletilmiş taslak, tam kaynakçalı (n=25 literatür taraması entegre
edilmiştir). Hedef konferans/dergi belirlenince biçim (şablon, sayfa sınırı,
atıf stili) o venue'ye uyarlanacaktır; içerik ve sayısal sonuçlar
değişmeyecektir. İkinci bir eğitim seed'i (seed=1) ile tam 8-konfigürasyonluk
doğrulama koşusu **tamamlanmıştır**; bulgular n=2 (seed=0, seed=1) üzerinden
raporlanmaktadır (bkz. §5.2, §5.3, §7). Bu doğrulama, çalışmanın en önemli
ikinci katkısını ortaya çıkarmıştır: n=1'de gözlenen "temiz çarpımsal" gecikme
deseni ikinci seed'te **tekrarlanmamış** — özellikle off-policy yapının tek
başına kararlılığı artırdığı iddiası yön değiştirmiştir — ancak GSAC'ın her
iki seed'de de dört yöntem içinde en dayanıklısı olması **tutarlı kalmıştır**.

---

## Özet

Bu çalışma, analitik (diferansiyellenebilir-simülasyon üzerinden hesaplanan)
politika gradyanı kullanan pekiştirmeli öğrenme yöntemlerinin **kaos
sınırındaki kırılma noktasını** — eğitimin sayısal olarak ıraksadığı ilk
episode'u — kaotik bir akışkanlar dinamiği kontrol görevinde nicel olarak
karakterize eder. *Nature*'da yayımlanan HydroGym platform makalesi [1], 61'den fazla
doğrulanmış akış kontrolü ortamı sunan, çözücüden bağımsız bir pekiştirmeli
öğrenme (RL) platformu tanımlar ve JAX tabanlı diferansiyellenebilir
Kolmogorov akışı ortamında **Gradyan-Zenginleştirilmiş PPO (GPPO)**'nun —
politika parametrelerine göre ödül yörüngesinin analitik gradyanını doğrudan
akışkanlar dinamiği çözücüsü üzerinden geri-yayılımla hesaplayıp standart PPO
gradyanıyla birleştiren bir yöntemin — eğitim iterasyonlarını standart PPO'ya
kıyasla **en az %65 azalttığını** raporlar. Ancak bu sonuç, makalenin
Kolmogorov akışı için tanımladığı geniş Reynolds aralığının (Re=40–500)
**yalnızca tek bir noktasında, tek eğitim seed'iyle** gösterilmiştir; analitik
gradyan enjeksiyonunun akışın **düzenli (laminer)** ve **kaotik
(türbülanslı)** rejimlerinde nasıl farklı davrandığı ayrıca test
edilmemiştir. Bu boşluk, kaotik dinamiklerde geri-yayılan gradyanların sistem
Jacobian'ının spektrumu nedeniyle kararsızlaştığını gösteren genel bir
bulguyla [7] ve türbülansın Lyapunov üstelinin Reynolds sayısıyla birlikte
büyüdüğünü, dolayısıyla öngörü ufkunun Re ile daralttığını gösteren klasik bir
sonuçla [25] örtüşmektedir.

Bu çalışma, gerçek `hydrogym.jax.envs.kolmogorov.KolmogorovFlow` ortamı
üzerinde dört algoritmayı — PPO, GPPO, ve literatürde HydroGym'e daha önce
uygulanmamış bir çift olan SAC [8] ile onun analitik-gradyanla
zenginleştirilmiş hâli **GSAC** — düzenli (Re=45) ve kaotik (Re=300)
rejimlerde, 500 episode'luk bütçelerle, **iki bağımsız eğitim seed'i (seed=0,
seed=1)** üzerinden test eder. GSAC'ın tasarımı, temaslı rijit-gövde
görevlerinde kısa-ufuklu analitik gradyan kullanan SHAC [3] ile bu fikri
off-policy maksimum-entropi bir çerçeveye taşıyan SAPO [4]'nun kesişiminde
konumlanır. Re=45'te GPPO'nun örnekleme avantajı her iki seed'de de doğrulanır
(seed=0'da %63, seed=1'de %77 episode tasarrufu; makalenin bildirdiği %65'e
[1] her iki değer de yakın büyüklük mertebesindedir). Re=300'de ise dört
algoritma da eğitimin bir noktasında sayısal kararsızlığa (NaN) düşer, ancak
kırılma zamanı hem algoritmaya hem de **seed'e** göre çarpıcı biçimde
farklılaşır: seed=0'da PPO 12., SAC 36., GPPO 83., GSAC 141. episode'da çöker
(PPO'ya göre 3.0×, 6.9×, 11.8× gecikme); seed=1'de ise PPO 35., **SAC 6.**
(PPO'dan **daha erken**, 0.17×), GPPO 40., GSAC 63. episode'de çöker. İkinci
seed, n=1'de gözlenen "off-policy yapı tek başına kararlılığı artırır"
iddiasını **doğrulamamış**, hatta tersine çevirmiştir; buna karşın analitik
gradyan enjeksiyonunun gecikme yönü (GPPO>PPO, GSAC>SAC) ve **GSAC'ın dört
yöntem içinde her iki seed'de de en dayanıklısı olması** tutarlı kalmıştır.
Ham eğitim eğrilerinin incelenmesi, kırılmanın öncülsüz/ani bir olay olduğunu
ve kalıcı-çöküş öncesi kısa bir "hayatta kalma" penceresinin (yalnızca
seed=0'ın GSAC'ında gözlenen) bir replay-tamponu seyreltme mekanizmasıyla
nicel olarak açıklanabildiğini — ve bu mekanizmanın seed=1'in farklı sonucunu
da (tamponun küçük olması nedeniyle hayatta kalma olmaması) doğru
öngördüğünü göstermektedir (§5.3). Bulgular, analitik gradyan enjeksiyonunun
kaotik rejimde kararsızlığı ortadan kaldırmadığını, yalnızca **ertelediğini**
ve bu erteleme büyüklüğünün de seed'e bağlı olarak büyük varyans
gösterdiğini; buna karşın off-policy yapının bu erteleyici etkiyi yalnızca
analitik gradyanla **birlikte** güvenilir biçimde sağladığını, tek başına
güvenilir olmadığını ortaya koymaktadır.

**Anahtar Kelimeler:** pekiştirmeli öğrenme, diferansiyellenebilir simülasyon,
analitik politika gradyanı, akışkanlar dinamiği kontrolü, kaotik dinamikler,
PPO, SAC

### Abstract

*(Title: "Gradient at the Edge of Chaos: The Breaking Point of Analytic
Gradient Methods in HydroGym")* This work characterizes the **breaking
point at the edge of chaos** — the first episode at which training
numerically diverges — for analytic-gradient reinforcement learning methods
on a chaotic fluid-dynamics control task. The *Nature* paper introducing
HydroGym [1] reports that Gradient-Enhanced
PPO (GPPO) — which backpropagates an analytic reward-trajectory gradient
directly through the differentiable fluid solver and mixes it with the
standard PPO gradient — reduces training iterations on the differentiable
Kolmogorov-flow environment by at least 65% relative to standard PPO. This
result is demonstrated at a single point within the paper's tested Reynolds
range (Re=40–500) with a single training seed; whether the analytic-gradient
advantage holds uniformly across regular (laminar) and chaotic (turbulent)
regimes is left untested — notable given the well-documented tendency of
backpropagation-through-time to become ill-conditioned under chaotic
dynamics [7], and the classical result that the Lyapunov exponent of
turbulence grows with Reynolds number, shrinking the predictability
horizon [25]. We evaluate PPO, GPPO, SAC [8], and a novel gradient-enhanced
SAC variant (GSAC) — positioned at the intersection of short-horizon
analytic-gradient actor-critic methods [3] and off-policy maximum-entropy
analytic-gradient methods [4] — on the real `hydrogym.jax` Kolmogorov-flow
environment at a regular (Re=45) and a chaotic (Re=300) operating point, 500
training episodes per configuration, across **two independent training seeds
(seed=0, seed=1)**. At Re=45, GPPO's sample-efficiency advantage is
reproduced in both seeds (63% and 77% fewer episodes to reach PPO's best
reward, both in the same order of magnitude as the paper's reported 65%).
At Re=300, all four algorithms eventually diverge numerically (NaN), but the
episode at which divergence first occurs differs sharply by both algorithm
*and* seed: in seed=0, PPO breaks at episode 12, SAC at 36, GPPO at 83, and
GSAC at 141 (3.0×, 6.9×, and 11.8× delay relative to PPO); in seed=1, PPO
breaks at 35, **SAC at 6** (*earlier* than PPO, 0.17×), GPPO at 40, and GSAC
at 63. The second seed does **not** reproduce the n=1 claim that off-policy
training alone delays instability — that direction reverses — but the
delaying effect of analytic-gradient injection (GPPO>PPO, GSAC>SAC) and,
most importantly, **GSAC being the most robust of the four algorithms**
replicate in both seeds. Direct inspection of the raw training curves shows
that the break is a sudden, precursor-free event in both seeds, and that a
short post-trigger "survival window" before permanent collapse (observed
only for GSAC in seed=0) is quantitatively explained by a replay-buffer
dilution mechanism that also correctly predicts the *absence* of such a
window in seed=1 (§5.3). We conclude that analytic gradient injection
delays — but does not eliminate — instability under chaotic dynamics, that
the size of this delay varies substantially across seeds, and that
off-policy training only reliably contributes to this delay when combined
with analytic gradient injection, not on its own.

---

## 1. Giriş

Akışkanlar dinamiğinin aktif kontrolü — sürükleme azaltımı, karışım artırımı,
aşırı olay (extreme event) bastırma gibi hedeflerle — pekiştirmeli öğrenmenin
son yıllarda en aktif uygulama alanlarından biri hâline gelmiştir. Alanın
kurucu çalışması olan Rabault ve ark. [23], derin RL'nin bir silindir
arkasındaki periyodik girdap dökülmesini kontrol ederek Re=100'de %8
sürükleme azaltımı sağlayabildiğini ilk kez göstermiş; bu çizgi, türbülanslı
ayrılma kabarcığı kontrolü [10] ve gerçek zamanlı, durum-bağımlı türbülanslı
iz (wake) kontrolü [24] gibi çalışmalarla türbülans rejimine genişlemiştir.
HydroGym [1], bu dağınık kod tabanlarını tek, çözücüden bağımsız bir
platformda birleştirerek laminerden türbülansa 61'den fazla ortam sunar ve
LBM, FEM gibi klasik çözücülerin yanı sıra JAX [13] tabanlı, uçtan uca
**diferansiyellenebilir** bir çözücü de içerir — bu çözücü altyapısı,
JAX-CFD [14] ve JAX-Fluids [15,16] gibi öncü diferansiyellenebilir CFD
kütüphanelerinin doğrudan bir devamıdır (dikkat çekici olarak JAX-Fluids'in
yazarları Bezgin ve Buhendwa, HydroGym makalesinin de ortak yazarları
arasındadır). Benzer motivasyonla geliştirilen, PyTorch tabanlı ve tamamen
diferansiyellenebilir bir başka akış-kontrolü kıyaslama paketi olan
FluidGym [2] da bu ekosistemin metodolojik bir akrabasıdır.

Diferansiyellenebilirlik, standart model-serbest RL'nin ötesine geçip
**analitik politika gradyanlarını** — ödülün politika parametrelerine göre
türevini, örnekleme yerine doğrudan zincir kuralıyla, simülatörün kendisi
üzerinden hesaplama imkânı sunar. HydroGym makalesi bu imkânı GPPO ile
gösterir: analitik gradyan, standart PPO'nun olabilirlik-oranı tabanlı
gradyanıyla karıştırılır ve Kolmogorov akışı ortamında eğitim iterasyonlarını
en az %65 azalttığı, daha düşük eylem genliğine sahip daha verimli
kontrolcüler ürettiği raporlanır [1]. Ne var ki bu sonuç **tek bir çalışma
noktasında** gösterilmiş olup, makalenin Kolmogorov akışı için tanımladığı
geniş Re=40–500 aralığının düzenli ve kaotik uçlarında analitik gradyanın
davranışı ayrıca karşılaştırılmamıştır.

Bu boşluk, diferansiyellenebilir programlama literatüründe iyi bilinen bir
riskle örtüşür. Metz ve ark. [7], yinelenen haritalarda, sayısal fizik
simülasyonlarında ve öğrenilmiş optimize edicilerde ortak bir "kaos-kaynaklı
başarısızlık modu" göstermiş; bunu sistemin Jacobian'ının spektrumuna
bağlamıştır. Akışkanlar-dinamiği tarafında ise Boffetta ve Musacchio [25],
homojen-izotropik türbülansta en büyük Lyapunov üstelinin Reynolds sayısıyla
birlikte (dimensiyonel beklentiden daha hızlı) büyüdüğünü ve öngörü ufkunun
bu üstelle ters orantılı olarak daraldığını göstermiştir. Bu iki bağımsız
teorik dayanak (biri optimizasyon, biri akışkanlar-dinamiği kökenli), kaotik
bir rejimde analitik gradyanın normunun/yönünün kelebek etkisiyle aşırı
duyarlı hâle gelmesini ve GPPO'nun bildirilen avantajının Re arttıkça
zayıflayabileceğini, hatta tersine dönebileceğini düşündürür.

Analitik gradyan enjeksiyonu şimdiye dek ağırlıklı olarak **on-policy**
yöntemlerle (GPPO gibi) veya rijit-gövde/temaslı robotik görevlerinde
uygulanmıştır: SHAC [3], kısa-ufuklu (short-horizon) rollout'lar ve
öğrenilmiş bir terminal değer fonksiyonuyla patlayan/kaybolan gradyan
sorununu hafifletir; AHAC [5] bu fikri, ufku sabit tutmak yerine sert (stiff)
dinamiklerden kaçınacak şekilde **adaptif olarak** ayarlayarak bir adım öteye
taşır. Off-policy tarafta ise SAPO [4], analitik-gradyan RL'yi maksimum-entropi
(SAC-benzeri) bir çerçeveye taşıyarak deforme-olabilir cisim görevlerinde
üstün performans göstermiştir — ancak bu yaklaşım hiç akışkanlar dinamiğine
uygulanmamıştır. Bu çalışma, SAC [8]'ın analitik gradyanla zenginleştirilmiş
bir versiyonu olan **GSAC**'ı dördüncü yöntem olarak ekleyerek, GPPO (SHAC
ailesi) ile SAPO ailesinin kesişimini akışkanlar dinamiğine taşır ve
off-policy replay-buffer yapısının kaotik rejimdeki gradyan kararsızlığını
on-policy GPPO'dan daha iyi tolere edip etmediğini sınar.

**Katkılar:**
1. HydroGym'in Kolmogorov akışı üzerindeki GPPO-vs-PPO sonucunu [1], tek bir
   Re noktasından düzenli–kaotik bir rejim karşılaştırmasına genişletiyoruz.
2. Literatürde HydroGym'e hiç uygulanmamış bir GSAC (SAC + analitik gradyan,
   SAPO [4] ailesinin akışkanlar dinamiğine ilk taşınması) varyantı sunuyor
   ve kararlılık avantajının kaynağının "analitik gradyan" mı yoksa
   "off-policy eğitim yapısı" mı olduğunu ilk kez ayrıştırıyoruz.
3. Kaotik rejimde dört algoritmanın da sayısal olarak ıraksadığını, ama
   ıraksama zamanının hem algoritmaya hem de eğitim seed'ine göre büyük
   ölçüde değiştiğini (12–141 episode aralığında) gösteriyoruz. **İki
   bağımsız seed'in** doğrudan karşılaştırılması, n=1'de gözlenen "temiz
   çarpımsal katkı" deseninin bir yapıt olduğunu ortaya koyuyor: analitik
   gradyanın geciktirici etkisi yön olarak her iki seed'de de tutarlı
   kalırken (GPPO>PPO, GSAC>SAC), off-policy eğitim yapısının **tek başına**
   geciktirici olduğu iddiası seed=1'de tersine dönüyor (SAC, PPO'dan daha
   erken çöküyor). Tek istikrarlı bulgu, **GSAC'ın dört yöntem içinde her iki
   seed'de de en dayanıklısı olması**dır.
4. Ham eğitim eğrilerini doğrudan inceleyerek kırılmanın mekanizmasını
   karakterize ediyoruz: kırılma öncesinde ödül/gradyan sinyalinde görünür
   bir öncül yoktur (ani, stokastik bir tetiklenme, her iki seed'de de
   gözlenir); on-policy yöntemlerde tetiklenme kalıcı çöküşle aynı episode'da
   gerçekleşirken, off-policy yöntemlerde replay tamponunun zehirli
   geçişleri seyreltmesi birkaç episode'luk bir kalıcı-çöküş ertelemesi
   yaratabilir — ve bu mekanizmadan türetilen nicel tahmin, seed=0'da GSAC'ta
   gözlenen 2-episode'luk ertelemeyi **ve** seed=1'de hiçbir algoritmada
   erteleme gözlenmemesini **her ikisini de** doğru öngörür (§5.3).

**Hipotez:** Analitik gradyanın normu/varyansı düşük Re'de kararlı, yüksek
Re'de kararsız olacak; bu kararsızlığa karşı GSAC'ın off-policy yapısı,
GPPO'nun on-policy yapısından daha dayanıklı kalacaktır. **(n=2 sonucu:**
GSAC'ın GPPO'dan daha dayanıklı olması her iki seed'de de doğrulanmıştır
(seed=0: 141 vs 83; seed=1: 63 vs 40); ancak bunun nedeni "off-policy yapı
tek başına daha dayanıklı" değil, off-policy yapının yalnızca analitik
gradyanla **birlikte** güvenilir bir katkı sağlamasıdır — bkz. §5.2.**)

## 2. İlgili Çalışmalar

**2.1 RL ile akış kontrolü ve HydroGym.** HydroGym [1], farklı çözücüleri tek
bir RL arayüzü altında birleştirerek laminerden türbülansa 61+ ortam sunar;
öğrenilen kontrol politikalarının ortamlar arasında sıfır-atış transfer
edilebildiğini (ör. 3B kanat kesitinde %38 sürükleme azaltımı) ve GPPO ile
Kolmogorov akışında ≥%65 örnekleme kazancı sağlandığını gösterir. FluidGym [2],
metodolojik olarak akraba, PyTorch-tabanlı ve tamamen diferansiyellenebilir
bir başka akış-kontrolü kıyaslama paketidir; PPO/SAC temel çizgileriyle
sunulmuştur. Bu çalışma, HydroGym'in bildirdiği kazanımın Reynolds sayısına
duyarlılığını inceler.

**2.2 Diferansiyellenebilir simülasyon üzerinden analitik politika
gradyanları.** GPPO'nun ait olduğu daha geniş yöntem ailesi, önce robotik/
kontrol literatüründe ortaya çıkmıştır. SHAC [3], kısa-ufuklu rollout'lar ve
pürüzsüz bir kritik fonksiyonuyla analitik gradyanı tam episode ufkundan
ayırarak patlayan/kaybolan gradyan sorununu hafifletir; muscle-actuated
lokomosyon görevlerinde 17 kattan fazla eğitim-süresi azalması bildirir —
GPPO/GSAC'ta kullanılan 6-adımlık kısa ufuk tasarımı bu yaklaşımın doğrudan
bir uyarlamasıdır. AHAC [5], sabit kısa-ufuk yerine ufku sert dinamiklerden
kaçınacak şekilde adaptif ayarlayarak lokomosyon görevlerinde model-serbest
temel çizgilere göre %40 daha fazla ödül elde eder. SAPO [4], bu fikri
maksimum-entropi off-policy bir çerçeveye taşıyarak, entropi
düzenlileştirmesinin analitik-gradyan optimizasyon yüzeyini
pürüzsüzleştirebileceği hipotezini rijit-cisim + deforme-olabilir malzeme
görevlerinde test eder. Bu çalışmadaki GSAC, SAPO'nun temel fikrini
akışkanlar dinamiğine taşıyan ilk uygulamadır.

**2.3 Kaotik sistemlerde analitik gradyan kararsızlığı.** Hipotezimizin
teorik dayanağı Metz ve ark. [7]: kaotik dinamik sistemlerde geri yayılan
gradyanların, sistem Jacobian'ının spektrumundan kaynaklanan bir
kelebek-etkisi mekanizmasıyla patladığını gösterir ve kırpma, kısaltılmış
BPTT veya gradyansız yöntemlere geçişi çözüm olarak önerir. Bu genel
gözlemin akışkanlar-dinamiği karşılığı Boffetta ve Musacchio [25]:
homojen-izotropik türbülansta en büyük Lyapunov üstelinin Reynolds sayısıyla
büyüdüğünü ve öngörü ufkunun bu üstelle ters orantılı olarak daraldığını
gösterir. Bu iki kaynak birlikte, 500-episode sonuçlarımızdaki "GPPO ve GSAC
de sonunda NaN'a düşüyor, ama çok daha geç" bulgusunun teorik zeminini
oluşturur: kararsızlık kaçınılmazdır, ayrışan şey yalnızca gecikme
mekanizmasıdır (off-policy yapı + kısa ufuk).

**2.4 Analitik gradyan mı, zeroth-order mi? Kısmi diferansiyellenebilirlik.**
Suh ve ark. [20], analitik ve zeroth-order politika gradyanlarını sertlik ve
süreksizlik içeren sistemlerde karşılaştırır ve önyargı-varyans (bias-variance)
analiziyle bir α-dereceli karma kestirici önerir — GSAC'taki %50-%50 karışım
tasarımının kabaca özel bir hâlidir. Bu çalışmanın 2026 tarihli bir takip
analizi [21], daha hafif bir hibrit-geçiş testi (DDCG) önererek aynı soruyu
güncel deneylerle yeniden sınar. ABPT [22], ödülün yalnızca kısmen
diferansiyellenebilir olduğu (quadrotor uçuş görevleri gibi) durumlar için
0-adım ve N-adım dönüşlerini birleştiren düzeltilmiş bir BPTT şeması önerir —
HydroGym'in kontrol-ödül ile fizik-durumu karışık yapısına yakın bir problem
sınıfıdır.

**2.5 Temel pekiştirmeli öğrenme algoritmaları.** Karşılaştırmanın iki
gradyansız çıpası: Schulman ve ark. [6]'nın PPO'su ve Haarnoja ve ark.
[8,9]'ın Soft Actor-Critic'i. PPO'nun avantaj kestirimi, TRPO [17] ve
Genelleştirilmiş Avantaj Kestirimi (GAE) [18]'e dayanır; SAC'ın
maksimum-entropi çerçevesinin kökeni olan Soft Q-learning [19], entropi
düzenlileştirmesinin off-policy öğrenmede neden GSAC'ın kaotik rejimde daha
dayanıklı kaldığını açıklayan teorik zemini sağlar: entropi terimi,
politikayı belirsizlik altında daha geniş bir eylem dağılımına zorlayarak tek
bir kötü-koşullu gradyan adımının etkisini sönümler.

**2.6 Akışkanlar dinamiğinde RL tabanlı aktif akış kontrolü (bağlam).**
Rabault ve ark. [23]'ın kurucu çalışmasından beri, gradyansız RL'nin
türbülanslı akış kontrolünde başarılı olduğu tekrar tekrar gösterilmiştir:
türbülanslı ayrılma kabarcığı kontrolü [10] klasik periyodik zorlamaya göre
%9'a karşı %6.8 sürükleme/ayrılma-alanı azaltımı sağlar; erken bir çalışma
[11] DRL'nin türbülanslı koşullara uygulanabilirliğini gösterir; gerçek
zamanlı, durum-bağımlı bluff-body iz kontrolü [24] rüzgar tüneli
deneylerinde model-tabanlı temel çizgilere göre 2–4× performans elde eder.
Bu kaynaklar akış kontrolünde RL'nin olgunluğunu doğrular ama hiçbiri
analitik gradyan enjeksiyonunu kaotik/türbülanslı rejimle sistematik olarak
karşılaştırmaz — bu çalışmanın hedeflediği boşluk tam olarak burasıdır.

## 3. Yöntem

### 3.1 Ortam

Deneyler, gerçek `hydrogym.jax.envs.kolmogorov.KolmogorovFlow` ortamında
yürütülmüştür: 64×64 ızgara (4096 serbestlik derecesi), pseudo-spektral
IMEX-RK4CN zaman entegrasyonu [1,13–16], 4 aktüatör modu (düşük dalga sayılı
sinüs kuvvetleme paternleri). Gözlem, hızın 8×8 alt-örneklenmiş büyüklüğüdür.
Bu çalışma, kütüphanenin resmi GPPO uygulamasını değil, aynı mekanizmayı
(§3.3) kendi PPO/SAC eğitim döngümüz içinde yeniden uygulayan bağımsız bir
kod tabanı kullanır; makalenin GPPO için kullandığı tam karışım oranı, ufuk
uzunluğu ve hiperparametreler yayımlanmış metinde/şekil başlıklarında
verilmediğinden, §3.3–3.4'te tanımlanan seçimler **kendi
yeniden-uygulamamıza** aittir ve makaleyle birebir aynı olduğu iddia
edilmemektedir.

Kütüphanenin `step_env` fonksiyonu, çağrı sırasında geçirilen
`params.dt`/`params.action_time` değerlerini yok sayıp bunun yerine
`self.default_params`'ı tekrar okur; bu yüzden zaman adımı ve eylem süresini
POC ölçeğine indirmek için `default_params` özelliğini override eden bir alt
sınıf (`FastKolmogorovFlow`) yazılmıştır. Kütüphanenin varsayılan
`reward_alpha=0` değeri ödülü yalnızca kontrol maliyetine indirger; bu
çalışmada `reward_alpha=1` kullanılarak akış enerjisi de ödüle dahil
edilmiştir.

**Reynolds sayısı seçimi.** Bu grid/dt kombinasyonunda (64×64, dt=0.001),
kütüphanenin kendi çözücüsünün — sıfır eylemle bile, yalnızca analitik bir
başlangıç koşulundan — ampirik olarak Re≈350–380 civarında sayısal olarak
ıraksadığı gözlemlenmiştir. Bu nedenle "kaotik" rejim için orijinal hedef olan
Re=450 yerine, bu sınırın altında kalan ve Re=45'le anlamlı bir kontrast
oluşturan en yüksek değer olan **Re=300** kullanılmıştır.

### 3.2 PPO ve GAE

Politika, ortalama eylem $\mu_\theta(s)$ üreten bir MLP (2 gizli katman, 64
birim, tanh) ve durumdan bağımsız, öğrenilebilir bir $\log\sigma$ ile
parametrelenmiş diyagonal Gauss dağılımıdır: $a \sim
\mathcal{N}(\mu_\theta(s), \sigma^2)$. Avantaj kestirimi GAE [18] ile
yapılır:

$$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}, \qquad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

($\gamma=0.99$, $\lambda=0.95$). PPO'nun TRPO'nun [17] güven-bölgesi
fikrinden türeyen kırpılmış vekil amaç fonksiyonu [6]:

$$L^{\text{PPO}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t(\theta)\hat{A}_t,\;
\text{clip}(\rho_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\,\hat{A}_t\big)\Big],
\quad \rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

($\epsilon=0.2$). Toplam kayıp, değer kaybı ve entropi bonusuyla
düzenlileştirilir: $L = -L^{\text{PPO}} + 0.5\,L^{V} - 0.001\,\mathcal{H}$.

### 3.3 GPPO: Analitik Gradyan Enjeksiyonu

SHAC [3] ve AHAC [5]'in kısa-ufuk fikrini izleyerek, $H=6$ adımlık kısa bir
ufukta, politikanın **ortalama** eylemi (örnekleme yok, tüm zincir
düzgün/pürüzsüz kalsın diye) kullanılarak diferansiyellenebilir simülatör
üzerinden ileri yayılım yapılır ve toplam ödülün negatifi $\theta$'ya göre
`jax.grad` ile analitik olarak türevlenir:

$$\mathcal{L}_{\text{analitik}}(\theta) = -\sum_{t=1}^{H} r\big(s_t,
\mu_\theta(s_t)\big), \qquad g_{\text{analitik}} =
\nabla_\theta \mathcal{L}_{\text{analitik}}(\theta)$$

Nihai gradyan, standart PPO gradyanı $g_{\text{PPO}} = \nabla_\theta
L(\theta)$ ile eşit ağırlıkla karıştırılır: $g_{\text{final}} = 0.5\,
g_{\text{analitik}} + 0.5\, g_{\text{PPO}}$ — Suh ve ark. [20]'ın önerdiği
α-dereceli karma kestiricinin ($\alpha=0.5$) özel bir hâli. Bu, HydroGym
makalesinin GPPO'yu tanımlayışıyla (politika parametrelerine göre ödül
yörüngesinin analitik gradyanının deterministik çözücü üzerinden
geri-yayılımla hesaplanıp standart gradyanla birleştirilmesi [1]) aynı
mekanizmadır; ufuk uzunluğu ve karışım ağırlığı bu çalışmanın kendi
seçimidir (bkz. §3.1, §7).

### 3.4 SAC ve GSAC

SAC [8,9], sıkıştırılmış (tanh) Gauss aktör, ikiz Q-kritik ve hedef ağlarla,
Soft Q-learning [19]'in kökenlendirdiği maksimum-entropi hedefini optimize
eder:

$$J(\pi) = \mathbb{E}\Big[\textstyle\sum_t r(s_t,a_t) + \alpha\,
\mathcal{H}(\pi(\cdot|s_t))\Big], \qquad
J_\pi(\phi) = \mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi_\phi}\big[\alpha\log
\pi_\phi(a|s) - \min(Q_{\psi_1}, Q_{\psi_2})(s,a)\big]$$

Kritikler, yumuşak hedef ağlarla ($\tau=0.005$) güncellenir. **GSAC**, SAPO
[4]'nun temel motivasyonunu (off-policy + maksimum-entropi + analitik
gradyan) izleyerek, GPPO ile aynı mekanizmayı (§3.3, $H=6$, deterministik
eylem) aktörün gradyanına uygular: $g_{\text{final}} = 0.5\,
g_{\text{analitik}} + 0.5\, g_{\pi}$, burada $g_\pi = \nabla_\phi
J_\pi(\phi)$. GSAC, bu çalışmada SAPO ailesinin akışkanlar dinamiğine ilk
taşınmasıdır; SAPO'nun kendisinden farklı olarak rijit/deforme-cisim
Rewarped platformu yerine HydroGym'in Kolmogorov akışı ortamını, ve SAPO'nun
kendi entropi-tabanlı pürüzsüzleştirme şeması yerine sabit %50-%50 karışımı
kullanır (bkz. §7).

## 4. Deneysel Kurulum

**Table 1.** Experimental configuration.

| Parameter | Value |
|---|---|
| Grid | 64×64 (4096 DOF) |
| Solver | pseudo-spectral IMEX-RK4CN (real `hydrogym.jax`) |
| Re (regular / chaotic) | 45 / 300 |
| Episodes / steps per episode | 500 / 15 |
| Analytic horizon (GPPO/GSAC) | $H=6$ steps |
| Gradient mixing ratio | 50% standard + 50% analytic |
| Learning rate | $3\times10^{-4}$ (Adam) |
| Network architecture | 2×64-unit MLP, tanh |
| Seeds | n=2 (seed=0, seed=1), see §5.2, §5.3, §7 |
| Hardware | RTX 5060 laptop GPU, JAX + CUDA 12 |

Tüm dört algoritma, her iki Re rejiminde, HydroGym makalesinin bildirdiği
eğitim bütçesiyle eşleşecek şekilde 500 episode ile eğitilmiştir.

## 5. Bulgular

### 5.1 Re=45 (düzenli akış): örnekleme verimliliği

GPPO, PPO'nun 500 episode boyunca ulaştığı en iyi düzeltilmiş (smoothed) ödül
seviyesine seed=0'da **episode 162**'de ulaşırken, PPO aynı seviyeye ancak
**episode 438**'de ulaşmıştır — **%63 episode tasarrufu**, makalenin
Kolmogorov akışı için bildirdiği **%65**'e [1] çarpıcı derecede yakın.
Seed=1'de aynı karşılaştırma **%77 episode tasarrufu** vermiştir (GPPO ep50,
PPO ep221) — mutlak sayılar farklı olsa da, GPPO'nun PPO'ya göre örnekleme
avantajının **yönü** her iki seed'de de tutarlıdır ve bu, kendi
yeniden-uygulamamızın makaledeki mekanizmayı (§3.3) doğru yakaladığının güçlü
bir işaretidir.

**Table 2.** Sample-efficiency and final-performance comparison at Re=45 (seed=0 and seed=1).

| Algorithm | Episode reaching PPO's best smoothed reward (seed=0 / seed=1) | Episode savings vs. PPO (seed=0 / seed=1) | Final reward, last-5-ep. mean (seed=0 / seed=1) |
|---|---|---|---|
| PPO  | 438 / 221 | — (reference) | −25.1 / −25.2 |
| GPPO | 162 / 50  | **63% / 77%** | −25.4 / −24.6 |
| SAC  | reached earlier (off-policy baseline) | — | **−16.7 / −16.9** |
| GSAC | reached earlier (off-policy baseline) | — | −18.0 / −17.1 |

*Figure 1* (`all4_500ep_comparison.png`, left panel) shows the corresponding
training curves for all four algorithms at Re=45 (seed=0).

SAC/GSAC'ta ise (off-policy replay-buffer yapısı zaten çok daha verimli
olduğundan — Tablo 2'de görüldüğü gibi son-5-episode ortalama ödül her iki
seed'de de SAC/GSAC için PPO/GPPO'dan belirgin biçimde daha iyi kalmaktadır)
GPPO-tarzı bir örnekleme-hızlandırma etkisi gözlenmemiştir: **GSAC, SAC'ın
nihai performansını bu ayarlarda hiçbir seed'de geçememiştir.**

### 5.2 Re=300 (kaotik akış): kırılma noktası dört algoritmada da var, ama seed'e göre çok farklı bir desende

Kaotik rejimde **dört algoritmanın da bir kırılma noktası vardır** — hepsi
500 episode'u kararlı tamamlayamamış, er ya da geç sayısal olarak
ıraksamıştır (NaN). Bu kırılma noktasına kadar geçen süre algoritmaya göre
çarpıcı biçimde farklılaşmaktadır — ama bu çalışmanın en önemli bulgusu, bu
farkın **seed'e göre de** çarpıcı biçimde değiştiğidir:

**Table 3.** Breaking point at Re=300, seed=0 vs. seed=1. See also *Figure 5*
(`seed_comparison.png`), *Figure 2* (`stability_delay_500ep.png`, seed=0)
and *Figure 1* (`all4_500ep_comparison.png`, right panel, seed=0).

| Algorithm | Breaking point, seed=0 | Delay vs. PPO, seed=0 | Breaking point, seed=1 | Delay vs. PPO, seed=1 |
|---|---|---|---|---|
| PPO  | 12  | 1.0× (reference) | 35 | 1.0× (reference) |
| SAC  | 36  | 3.0× | **6**  | **0.17× (earlier than PPO)** |
| GPPO | 83  | 6.9× | 40 | 1.14× |
| GSAC | 141 | 11.8× | 63 | 1.8× |

Seed=0'da bu tablo neredeyse çarpımsal bir yapı sergilemekteydi: off-policy
yapı tek başına (SAC/PPO) 3.0× gecikme sağlarken, analitik gradyan tek
başına on-policy zeminde (GPPO/PPO) 6.9×, off-policy zeminde (GSAC/SAC) 3.9×
gecikme sağlıyor, ikisi birlikte (GSAC) 11.8×'e ulaşıyordu (3.0×3.9≈11.7).
**Bu desen seed=1'de tekrarlanmamıştır.** Seed=1'de SAC, PPO'dan **daha
erken** (episode 6'da, PPO'nun episode 35'ine karşı) çökmüştür — off-policy
yapının tek başına kararlılığı artırdığı iddiası burada **tersine
dönmektedir**. Buna karşın iki yön tutarlı kalmaktadır:

1. **Analitik gradyan enjeksiyonu her iki seed'de de geciktiricidir:**
   GPPO, PPO'dan daha geç kırılır (6.9× ve 1.14×, her ikisi de >1); GSAC, SAC'tan
   daha geç kırılır (3.9× ve 10.5×, her ikisi de >1) — büyüklük çok değişse de
   yön değişmemektedir.
2. **GSAC, dört yöntem içinde her iki seed'de de en dayanıklısıdır** (seed=0:
   141 episode; seed=1: 63 episode — ikisi de kendi seed'indeki en yüksek
   değer). Bu, tekil bir mekanizmadan değil, iki farklı geciktirici etkinin
   (analitik gradyan + off-policy yapı, yalnızca birlikte güvenilir biçimde)
   üst üste binmesinden kaynaklanıyor olabilir; bunun ayrıntılı mekanik
   açıklaması §5.3'te sunulmaktadır.

Kısacası: n=1'in önerdiği "temiz, çarpımsal, iki bağımsız mekanizma" hikâyesi
fazla iyimserdi. n=2 ile ortaya çıkan daha gerçekçi ve daha ilginç bulgu,
off-policy yapının kararlılığa katkısının **kendi başına güvenilir
olmadığı**, yalnızca analitik gradyanla **birleştiğinde** tutarlı bir fayda
sağladığıdır.

### 5.3 Kırılmanın mekanizması: ani tetiklenme ve tampon-seyreltme etkisi

Yukarıdaki tablo *ne zaman* kırıldığını gösteriyor; bu alt bölüm, elimizdeki
ham eğitim eğrilerini (`episode_rewards`, `grad_norms`) doğrudan inceleyerek
*neden* kırıldığına dair iki somut bulgu sunar.

**(a) Kırılma önce sinyal vermeden, ani gerçekleşiyor.** Her dört
algoritma için de, **her iki seed'de de**, kırılma öncesi 40 episode'luk
pencerede ödül ortalaması ve standart sapması kararlıdır — hiçbir
algoritmada/seed'de kırılmaya doğru bir "ısınma" (ramp) veya kademeli
bozulma görülmez:

**Table 4.** Pre-break reward statistics (40-episode window immediately
before the breaking point, seed=0 and seed=1). No ramp or gradual
degradation precedes divergence in any algorithm or seed.

| Algorithm | First-half mean (seed=0 / seed=1) | Second-half mean (seed=0 / seed=1) | Std. dev. (seed=0 / seed=1) |
|---|---|---|---|
| PPO  | −32.75 / −32.28 | −32.77 / −33.05 | 1.35 / 1.32 |
| SAC  | −33.31 / −32.76 | −32.50 / −32.79 | 1.30 / 1.39 |
| GPPO | −32.61 / −32.38 | −32.98 / −33.01 | 1.10 / 1.24 |
| GSAC | −32.49 / −32.19 | −32.46 / −32.08 | 1.32 / 1.16 |

Bu bulgu — kırılmanın öncülsüz, ani bir olay olması — **her iki seed'de de**
gözlenmiştir; §5.2'de bulunan seed-bağımlı varyansın aksine, bu nitelik
tamamen tutarlıdır.

GPPO/GSAC'ın analitik gradyan normlarında da (kırılmadan hemen önceki 15
episode, bkz. `results_500_gppo_high.json`/`results_500_gsac_high.json`)
benzer bir düz seyir gözlenir (~20–30 aralığında dalgalanma, belirgin bir
artış trendi yok). Bu, kırılmanın politikanın *kademeli olarak* kontrol
kuvvetini artırıp çözücüyü zorlamasından değil — reward/gradyan sinyalinde
görünmeyen, **tek bir episode'un rastgele örneklenen aksiyon dizisinin**,
o an ki politika ile birleşince akışı çözücünün Re≈350–380 civarındaki
sayısal kararlılık sınırının (bkz. §3.1) ötesine iten, ani ve stokastik bir
olaydan kaynaklandığını düşündürür — Metz ve ark. [7]'ın kaos-kaynaklı
gradyan patlamasının ve Boffetta-Musacchio [25]'nun Re ile daralan öngörü
ufkunun beklediği türden bir "kelebek etkisi" tetiklenmesiyle tutarlıdır.

**(b) On-policy ve off-policy algoritmalar tetiklenme sonrası farklı davranır
— tampon-seyreltme etkisi.** Tetiklenme anıyla *kalıcı* çöküş anı arasındaki
mesafeyi inceleyince net bir ayrım ortaya çıkar:

**Table 5.** Trigger episode vs. permanent-collapse episode, seed=0 and
seed=1. The "recovered episodes" column counts intermediate episodes with a
valid (non-NaN) reward between the trigger and the permanent collapse — for
GSAC in seed=0, episodes 142 and 143 recover before episode 144 collapses
for good; in seed=1, no algorithm shows a recovery window (see §5.3(c) for
why this is exactly what the dilution mechanism predicts). See also
*Figure 3* (`mechanism_analysis.png`, left panel, seed=0).

| Algorithm | Trigger (seed=0) | Collapse (seed=0) | Recovered (seed=0) | Trigger (seed=1) | Collapse (seed=1) | Recovered (seed=1) |
|---|---|---|---|---|---|---|
| PPO  | ep 12  | ep 12  | 0 | ep 35 | ep 35 | 0 |
| SAC  | ep 36  | ep 36  | 0 | ep 6  | ep 6  | 0 |
| GPPO | ep 83  | ep 83  | 0 | ep 40 | ep 40 | 0 |
| GSAC | ep 141 | ep 144 | **2** (ep 142, ep 143) | ep 63 | ep 63 | 0 |

PPO/GPPO'da (on-policy), o episode'un rollout'u zaten o episode'un kendi
gradyan güncellemesinde kullanıldığı için tetiklenme ve kalıcı çöküş
**aynı anda** olur — kaçış yolu yoktur. SAC'ta da (off-policy, replay-buffer
kullanmasına rağmen) tetiklenme aynı episode'da kalıcılaşır, çünkü tetiklenme
episode 36'da gerçekleştiğinde tamponda yalnızca ~540 geçiş vardır ve bu
episode'un kendi ~15 "zehirli" geçişi tamponun **%2.8**'ini oluşturur; aynı
episode içinde 10 güncelleme × 64 örneklem = 640 çekilişte bu geçişlerden
hiçbirinin örneklenmeme olasılığı pratikte sıfırdır. GSAC'ta ise tetiklenme
çok daha geç (episode 141) gerçekleştiği için tampon çok daha büyüktür
(~2115 geçiş) ve aynı ~15 zehirli geçiş tamponun yalnızca **%0.7**'sini
oluşturur — bu, o episode'un 640 çekilişinde zehirli geçişlerden hiç
örneklenmeme olasılığını ~%1'e çıkarır ve gözlemlenen 2 episode'luk "hayatta
kalma" bu düşük ama sıfır-olmayan olasılıkla niteliksel olarak tutarlıdır.

**Table 6.** Replay-buffer dilution at the trigger episode (off-policy
algorithms only, seed=0 and seed=1). See also *Figure 3*
(`mechanism_analysis.png`, right panel, seed=0).

| Algorithm | Trigger (seed=0) | Buffer (seed=0) | Bad-fraction (seed=0) | Survival prob. (seed=0) | Trigger (seed=1) | Buffer (seed=1) | Bad-fraction (seed=1) | Survival prob. (seed=1) |
|---|---|---|---|---|---|---|---|---|
| SAC  | 36  | ~540 transitions  | 2.8% | ≈0% | 6  | ~90 transitions   | 16.7% | ≈0% |
| GSAC | 141 | ~2115 transitions | 0.7% | ≈1% | 63 | ~945 transitions  | 1.6%  | ≈0.004% |

**Yorum ve (c) seed=1 ile çapraz-doğrulama:** Bu mekanik model, off-policy/
analitik-gradyan gecikmesinin (§5.2) en azından bir kısmının "daha akıllı/
daha kararlı öğrenme" değil, **tamponun zehirli geçişleri seyreltmesi** gibi
oldukça mekanik bir nedene bağlı olabileceğini gösterir — tampon ne kadar
büyükse (yani tetiklenme eğitimde ne kadar geç olursa), kalıcı çöküşün
birkaç episode ertelenme şansı o kadar artar. Bu model, elimizde seed=0
verisiyle kurulduktan sonra **seed=1 üzerinde bağımsız olarak test
edilmiştir** ve model doğru öngörüde bulunmuştur: seed=1'de hem SAC'ın hem
de GSAC'ın tetiklenme anındaki tamponu (sırasıyla ~90 ve ~945 geçiş) seed=0'a
göre çok daha küçüktür — çünkü her iki algoritma da seed=1'de çok daha erken
tetiklenmiştir (§5.2) — ve buna bağlı olarak hesaplanan hayatta-kalma
olasılığı her ikisinde de ~%0'a yakındır; nitekim seed=1'de **hiçbir
algoritmada** bir kurtarma penceresi gözlenmemiştir (Tablo 5). Yani mekanizma
kendisi (büyük tampon → düşük zehirli-geçiş oranı → hayatta kalma şansı),
seed=0'da GSAC'ın 2-episode'luk kurtarmasını **ve** seed=1'de hiçbir
algoritmanın kurtarılamamasını **her ikisini de** doğru öngörmektedir — bu,
§5.2'deki ham gecikme çarpanlarının aksine, **n=2 ile nicel olarak
doğrulanmış** bir bulgudur.

**Açık soru (hâlâ açık kalan tek nokta):** Bu analiz *tetiklenme sonrası
kilitlenme* mekanizmasını açıklıyor ve iki seed arasında doğrulanmıştır, ama
*tetiklenme episode'unun kendisinin* neden algoritmaya ve seed'e göre bu denli
değiştiğini (ör. SAC'ın seed=0'da episode 36'da, seed=1'de ise episode 6'da
tetiklenmesi) açıklamıyor — bunun için eylem genliğinin episode-episode nasıl
evrildiğinin ayrıca loglanması gerekir (mevcut `episode_rewards`/
`grad_norms` çıktısı bunu içermiyor); bu, doğal bir sonraki tanı deneyi
olarak §7'de not edilmiştir.

## 6. Tartışma ve Katkı

Bu çalışma, HydroGym makalesinin Kolmogorov akışı üzerindeki GPPO-vs-PPO
sonucunu [1] iki yönde genişletir:

1. **Rejim genişlemesi:** Re=45'te makale her iki seed'de de yön olarak
   doğrulanır (%63 ve %77 vs makalenin bildirdiği %65); Re=300'de makalenin
   göstermediği bir bulgu ortaya çıkar — GPPO'nun avantajı "daha az
   episode'da daha iyi ödül"den "daha uzun süre kararlı kalabilme"ye
   dönüşür, ve bu avantaj da sonunda tükenir.
2. **Mekanizma ayrıştırması (GSAC katkısı, n=2 ile düzeltilmiş):** GSAC
   sayesinde, kararlılık avantajının "analitik gradyan" mı yoksa "off-policy
   eğitim yapısı" mı kaynaklı olduğu sorusu ilk kez ayrıştırılabilir. n=1
   ile ilk izlenim, ikisinin bağımsız ve yaklaşık çarpımsal katkılar
   sağladığı yönündeydi; **ikinci bir seed bu izlenimi düzeltmiştir.**
   Analitik gradyanın geciktirici etkisi (GPPO>PPO, GSAC>SAC) her iki
   seed'de de yön olarak tutarlı kalırken, off-policy yapının **tek başına**
   (SAC vs. PPO) geciktirici olduğu iddiası seed=1'de tersine dönmüştür —
   SAC, PPO'dan daha erken çökmüştür. Yalnızca **GSAC'ın dört yöntem
   içinde her iki seed'de de en dayanıklısı olması** tutarlı kalmıştır; bu,
   off-policy yapının kararlılığa katkısının kendi başına değil, analitik
   gradyanla birleştiğinde güvenilir olduğunu düşündürmektedir.

Bulgular, Metz ve ark. [7]'ın genel kaos-kaynaklı gradyan kararsızlığı
gözlemini ve Boffetta-Musacchio [25]'nun Re-bağımlı öngörülemezlik
sonucunu somut bir akışkanlar dinamiği kontrol görevine taşır. n=1 sonucu bu
olguyu "algoritma tasarımıyla azaltılabilir, iki bağımsız ve öngörülebilir
mekanizma" olarak çerçeveliyordu; n=2 ile ortaya çıkan daha gerçekçi tablo,
kaotik rejimdeki kararlılığın **hem algoritma tasarımına hem de eğitimin
rastgele seyrine (seed'e)** bağlı olduğunu, ve bu iki etkenin birbirinden
ayrıştırılmasının düşünüldüğünden daha zor olduğunu göstermektedir. Buna
karşın §5.3'teki tampon-seyreltme mekanizması, tetiklenme-sonrası davranışı
(kurtarma penceresi var mı, yok mu) her iki seed'de de doğru öngörerek,
kırılmanın *ne zaman kalıcı hale geldiği* sorusuna algoritma tasarımından
bağımsız, saf mekanik bir açıklama sunmaktadır — bu, n=2 ile nicel olarak
sınanmış ve doğrulanmış tek mekanizmik bulgudur. Bulgular, HydroGym
makalesinin ana iddiasını (GPPO'nun faydalı olduğu) çürütmez; tersine bunu
daha geniş bir algoritma ailesi, rejim uzayı ve **birden fazla seed**
üzerinde sınar ve kaotik rejimde kararlılığın kaynağının düşünüldüğünden
daha katmanlı ve seed-duyarlı olduğunu gösterir.

## 7. Sınırlamalar

- **n=2 seed (seed=0, seed=1)** — istatistiksel anlamda hâlâ küçük bir
  örneklem; iki seed, bir noktanın "rastgele mi yoksa güvenilir mi"
  olduğunu ayırt etmeye yeter, ama biçimsel bir anlamlılık testi (ör.
  permütasyon/bootstrap) için yetersizdir. n=2'nin asıl değeri, n=1 ile
  görülen bir deseni (çarpımsallık) çürütebilmesidir — nitekim burada da
  öyle olmuştur (§5.2). Daha fazla seed (3-5), özellikle SAC'ın seed=0 ile
  seed=1 arasında gösterdiği yön değişikliğinin (3.0× gecikme ↔ 0.17×
  erken çöküş) hangi ucun "tipik", hangisinin "aykırı" olduğunu ayırt etmek
  için gereklidir; bu çalışmanın kapsamı şimdilik n=2 ile sınırlandırılmıştır.
- **Re=300'deki gecikme çarpanları seed'e göre büyük varyans gösteriyor**
  (GPPO/PPO: 1.14×–6.9×; GSAC/SAC: 3.9×–10.5×; SAC/PPO: 0.17×–3.0×, yön
  değişimiyle) — bunlar "yasa" değil, iki nokta-gözlemdir. Tek istikrarlı
  bulgu GSAC'ın her iki seed'de de en dayanıklı algoritma olmasıdır;
  bileşenlerin (off-policy, analitik gradyan) bağımsız katkılarının kesin
  büyüklüğü açık kalmıştır.
- **Re=300, orijinal hedef Re=450 yerine kullanıldı** — kütüphanenin kendi
  çözücüsünün bu grid/dt kombinasyonunda Re≈350–380 civarında kararlılık
  sınırına sahip olduğu keşfedilmiştir.
- **GPPO/GSAC'ın karışım oranı, ufuk uzunluğu ve diğer hiperparametreleri
  kendi yeniden-uygulamamıza aittir** — HydroGym makalesi bu değerleri
  erişebildiğimiz metin/şekil başlıklarında paylaşmamaktadır; %63/%77'lik
  sonuçların %65 ile aynı büyüklük mertebesinde olması güçlü bir işaret olsa
  da, iki uygulamanın birebir aynı olduğu iddia edilmemektedir.
- **Kısa episode'lar (15 RL-adımı) ve kısa analitik ufuk (6 adım)** — hem RL
  yakınsaması hem kaotik ayrışmanın (Lyapunov büyümesi [25]) tam
  gözlemlenmesi için yetersiz olabilir.
- **GSAC'ın karışım ağırlığı sabit %50-%50** — Suh ve ark. [20]'ın önerdiği
  adaptif α-dereceli kestirici, AHAC [5]'in adaptif-ufuk fikri veya SAPO
  [4]'nun entropi-tabanlı pürüzsüzleştirmesi denenmemiştir; bunlar, Re=300'de
  kararsızlığı geciktirmek yerine tamamen önlemek için doğal bir sonraki
  adımı işaret etmektedir.
- **Tetiklenme-episode'unun kendisinin nedeni açık kalıyor (§5.3)** — bu,
  n=2 ile de çözülmeyen tek açık soru olarak kaldı. Mevcut kayıtlar
  (`episode_rewards`, `grad_norms`) yalnızca episode-seviyesi özet
  istatistikler içeriyor; eylem genliğinin episode-episode nasıl
  evrildiğini gösteren bir log tutulmamıştır. Doğal bir sonraki tanı deneyi,
  aynı seed/config'i (JAX PRNG tam determinize olduğundan bit-bit aynı
  şekilde) kısa bir ön-eşiğe kadar (ör. PPO için ilk 15, GPPO için ilk 90
  episode) eylem-normu loglayarak yeniden çalıştırmaktır — bu, henüz
  yapılmamıştır, ancak ucuz ve hızlı (birkaç dakikalık) bir takip
  deneyidir.
- **§5.3'teki tampon-seyreltme mekanizması artık n=2 ile sınanmıştır ve
  doğrulanmıştır** — seed=0'da kurulan model (büyük tampon → düşük
  zehirli-geçiş oranı → hayatta kalma şansı), seed=1'in tamamen farklı
  tetiklenme profilini (hem SAC hem GSAC çok daha erken tetiklendi, dolayısıyla
  çok daha küçük tamponla) doğru biçimde "kurtarma penceresi olmayacak" olarak
  öngörmüştür (Tablo 5, 6). Bu, çalışmanın n=2 ile gerçekten doğrulanmış tek
  mekanizmik iddiasıdır; yine de n=2, bu modelin genel geçerliliğini kesin
  olarak kanıtlamaz — yalnızca iki bağımsız gözlemle tutarlı olduğunu
  gösterir.

## 8. Sonuç ve Gelecek Çalışma

Analitik gradyan enjeksiyonu, kaotik rejimde sayısal kararsızlığı ortadan
kaldırmaz — **erteler**. İki bağımsız eğitim seed'i üzerinden test edildiğinde,
bu ertelemenin **yönü** (GPPO>PPO, GSAC>SAC) tutarlı kalırken, **büyüklüğü**
seed'e göre büyük ölçüde değişmektedir; off-policy eğitim yapısının tek
başına (SAC vs. PPO) koruyucu olduğu iddiası ise ikinci seed'te tersine
dönmüştür. Buna karşın, **GSAC'ın test edilen dört yöntem içinde her iki
seed'de de en dayanıklı konfigürasyon olması** istikrarlı kalmıştır — bu,
off-policy yapının kararlılığa katkısının yalnızca analitik gradyanla
**birlikte** güvenilir olduğunu düşündürmektedir. Ayrıca, kırılma sonrası
davranışı (kısa bir "hayatta kalma" penceresi olup olmayacağını) açıklayan
tampon-seyreltme mekanizması, seed=0 üzerinde kurulup seed=1 üzerinde
bağımsız olarak doğrulanmıştır — bu çalışmanın n=2 ile nicel olarak
sınanmış tek mekanizmik bulgusudur.

**Bu tur için kapsam kararı:** Kullanılan hesaplama bütçesi göz önüne
alındığında, bu taslak için n=2 (seed=0, seed=1) yeterli kabul edilmiş ve
üçüncü bir seed'e (seed=2) geçilmemiştir. Gelecekteki planlanan işler:

1. **3–5 seed'e genişletmek** ve biçimsel bir istatistiksel anlamlılık
   testi (permütasyon/bootstrap, küçük n için t-testinden daha uygun)
   uygulamak — özellikle SAC'ın seed=0/seed=1 arasında gözlenen yön
   değişikliğinin hangi ucun "tipik" olduğunu netleştirmek için.
2. **Tetiklenme-episode'unun kendisinin nedenini** açıklığa kavuşturmak için
   eylem-genliği loglu bir takip deneyi (§7).
3. GSAC'ın karışım ağırlığını AHAC [5]/SAPO [4] tarzı adaptif bir şemaya
   taşımak; Boffetta-Musacchio [25] çerçevesini kullanarak gecikme
   çarpanlarının Re'ye karşı nicel bir ölçeklenme yasasına oturup
   oturmadığını test etmek.

## Kaynaklar

[1] C. Lagemann, S. Mokbel, M. Gondrum, M. Rüttgers, Y. Wang, P. Suárez,
L. Paehler, D. A. Bezgin, A. B. Buhendwa, J. L. Callaham, S. Ahnert,
N. Zolman, X. Shao, J.-C. Loiseau, N. Adams, M. Meinke, W. Schröder,
K. Lagemann, E. Lagemann, R. Vinuesa, S. L. Brunton, "The HydroGym
Reinforcement Learning Platform for Fluid Dynamics," *Nature*, 2026.
doi:10.1038/s41586-026-10917-6. Ön baskı: arXiv:2512.17534.

[2] J. Becktepe, A. Franz, N. Thuerey, S. Peitz, "Plug-and-Play Benchmarking
of Reinforcement Learning Algorithms for Large-Scale Flow Control
(FluidGym)," arXiv:2601.15015, 2026.

[3] J. Xu, V. Makoviychuk, Y. Narang, F. Ramos, W. Matusik, A. Garg,
M. Macklin, "Accelerated Policy Learning with Parallel Differentiable
Simulation (SHAC)," in *Proc. ICLR*, 2022. arXiv:2204.07137.

[4] E. Xing, V. Luk, J. Oh, "Stabilizing Reinforcement Learning in
Differentiable Multiphysics Simulation (SAPO / Rewarped)," in *Proc. ICLR
(Spotlight)*, 2025. arXiv:2412.12089.

[5] I. Georgiev, K. Srinivasan, J. Xu, E. Heiden, A. Garg, "Adaptive Horizon
Actor-Critic for Policy Learning in Contact-Rich Differentiable Simulation
(AHAC)," in *Proc. ICML*, 2024. arXiv:2405.17784.

[6] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal
Policy Optimization Algorithms," arXiv:1707.06347, 2017.

[7] L. Metz, C. D. Freeman, S. S. Schoenholz, T. Kachman, "Gradients are Not
All You Need," arXiv:2111.05803, 2021.

[8] T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, "Soft Actor-Critic:
Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic
Actor," in *Proc. ICML*, 2018. arXiv:1801.01290.

[9] T. Haarnoja, A. Zhou, K. Hartikainen ve ark., "Soft Actor-Critic
Algorithms and Applications," arXiv:1812.05905, 2018.

[10] B. Font, F. Alcántara-Ávila, J. Rabault, R. Vinuesa, O. Lehmkuhl, "Deep
Reinforcement Learning for Active Flow Control in a Turbulent Separation
Bubble," *Nature Communications*, 2025. doi:10.1038/s41467-025-56408-6.

[11] "Applying Deep Reinforcement Learning to Active Flow Control in
Turbulent Conditions," arXiv:2006.10683, 2020.

[12] "Deep Reinforcement Learning for Robotics: A Survey of Real-World
Successes," *Annual Review of Control, Robotics, and Autonomous Systems*,
2025.

[13] J. Bradbury, R. Frostig, P. Hawkins ve ark., "JAX: Composable
Transformations of Python+NumPy Programs," 2018.
github.com/jax-ml/jax.

[14] D. Kochkov, J. A. Smith, A. Alieva, Q. Wang, M. P. Brenner, S. Hoyer,
"Machine Learning–Accelerated Computational Fluid Dynamics (JAX-CFD),"
*PNAS* 118(21), 2021. doi:10.1073/pnas.2101784118.

[15] D. A. Bezgin, A. B. Buhendwa, N. A. Adams, "JAX-Fluids: A
Fully-Differentiable High-Order CFD Solver for Compressible Two-Phase
Flows," arXiv:2203.13760, 2022.

[16] D. A. Bezgin, A. B. Buhendwa, N. A. Adams, "JAX-Fluids 2.0: Towards HPC
for Differentiable CFD of Compressible Two-Phase Flows,"
arXiv:2402.05193, 2024.

[17] J. Schulman, S. Levine, P. Moritz, M. I. Jordan, P. Abbeel, "Trust
Region Policy Optimization (TRPO)," arXiv:1502.05477, 2015.

[18] J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel,
"High-Dimensional Continuous Control Using Generalized Advantage
Estimation (GAE)," arXiv:1506.02438, 2015 (ICLR 2016'da sunulmuştur).

[19] T. Haarnoja, H. Tang, P. Abbeel, S. Levine, "Reinforcement Learning
with Deep Energy-Based Policies (Soft Q-Learning)," in *Proc. ICML*, 2017.
arXiv:1702.08165.

[20] H. J. T. Suh, M. Simchowitz, K. Zhang, R. Tedrake, "Do Differentiable
Simulators Give Better Policy Gradients?," in *Proc. ICML*, 2022.
arXiv:2202.00817.

[21] K. Onoda, P. Parmas, M. Yaguchi, Y. Matsuo, "Does 'Do Differentiable
Simulators Give Better Policy Gradients?' Give Better Policy Gradients?,"
arXiv:2604.18161, 2026.

[22] F. Li, F. Sun, T. Zhang, D. Zou, "ABPT: Amended Backpropagation through
Time with Partially Differentiable Rewards," arXiv:2501.14513, 2025.

[23] J. Rabault, M. Kuchta, A. Jensen, U. Réglade, N. Cerardi, "Artificial
Neural Networks Trained through Deep Reinforcement Learning Discover
Control Strategies for Active Flow Control," *Journal of Fluid Mechanics*
865, 281–302, 2019. doi:10.1017/jfm.2019.62.

[24] "Autonomous Real-Time Control of Turbulent Dynamics (REACT)," Imperial
College London, arXiv:2509.11002, 2026.

[25] G. Boffetta, S. Musacchio, "Chaos and Predictability of
Homogeneous-Isotropic Turbulence," *Physical Review Letters* 119, 054102,
2017. doi:10.1103/PhysRevLett.119.054102. arXiv:1707.01648.

*Kaynakça 26 Ağustos 2026 tarihinde gerçek zamanlı web taraması ile
doğrulanmıştır (başlık, yazar listesi ve yayın bilgisi her kaynağın kendi
sayfasından teyit edilmiştir); ayrıca bkz. tamamlayıcı belge
`HydroGym_Literatur_Taramasi_HaktanGuloglu.pdf` (N=25, 25 Ağustos 2026).*

## Şekiller ve Tablolar (mevcut, üretildi)

Tüm şekil başlıkları, eksen etiketleri ve tablo içerikleri İngilizce olarak
üretilmiştir (uluslararası okunabilirlik için); yalnızca çevreleyen düz metin
Türkçedir.

**Şekiller:**
- **Figure 1** — `all4_500ep_comparison.png`: left panel, Re=45, all 4
  algorithms, smoothed reward curves over 500 episodes (§5.1); right panel,
  Re=300, raw reward curves with first-NaN episode marked per algorithm
  (§5.2).
- **Figure 2** — `stability_delay_500ep.png`: bar chart of the first-NaN
  episode at Re=300 per algorithm, annotated with delay multiplier vs. PPO
  (§5.2, Table 3).
- **Figure 3** — `mechanism_analysis.png`: left panel, trigger episode vs.
  permanent-collapse episode per algorithm (§5.3, Table 5); right panel,
  replay-buffer bad-transition fraction at the trigger episode for the two
  off-policy algorithms (§5.3, Table 6).
- **Figure 4** — `paper_comparison_500ep.png`: PPO vs. GPPO, direct
  side-by-side format matching Figure 5 of the original HydroGym paper [1]
  (not to be confused with our own Figure 5 below), at both Re=45
  (reproducing the paper's regime) and Re=300 (our extension, untested in
  the original paper).
- **Figure 5** — `seed_comparison.png`: left panel, first-NaN episode at
  Re=300 for all 4 algorithms, seed=0 vs. seed=1, grouped bars; right panel,
  the same data expressed as delay multiplier vs. PPO — visually shows the
  SAC direction-reversal and GSAC's consistent top-robustness across both
  seeds (§5.2, Table 3).

**Tablolar:** Table 1 (§4, experimental configuration), Table 2 (§5.1,
sample efficiency and final performance at Re=45, seed=0 and seed=1),
Table 3 (§5.2, breaking point at Re=300, seed=0 and seed=1), Table 4 (§5.3,
pre-break reward statistics, seed=0 and seed=1), Table 5 (§5.3, trigger vs.
permanent-collapse episode, seed=0 and seed=1), Table 6 (§5.3, replay-buffer
dilution, seed=0 and seed=1, with cross-seed validation of the mechanism).
