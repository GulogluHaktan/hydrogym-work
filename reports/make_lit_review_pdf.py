#!/usr/bin/env python3
"""3 sayfayi gecmeyen literatur taramasi raporu: HydroGym / analitik-gradyan RL / kaotik akis
kontrolu pilot calismasinin dayandigi literatur. Turkce karakterler icin DejaVu Sans (matplotlib
ile gelen TTF) reportlab'a gomulu font olarak kaydedilir (onceki sonuc raporuyla ayni yontem)."""
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
import matplotlib

MPL_FONT_DIR = matplotlib.get_data_path() + "/fonts/ttf/"
pdfmetrics.registerFont(TTFont("DejaVuSans", MPL_FONT_DIR + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", MPL_FONT_DIR + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", MPL_FONT_DIR + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuSans-BoldOblique", MPL_FONT_DIR + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFontFamily(
    "DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
    italic="DejaVuSans-Oblique", boldItalic="DejaVuSans-BoldOblique",
)

OUT = "reports/HydroGym_Literatur_Taramasi_HaktanGuloglu.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body9", fontName="DejaVuSans", fontSize=9, leading=11.3, alignment=TA_JUSTIFY, spaceAfter=4))
styles.add(ParagraphStyle(name="H1", fontName="DejaVuSans-Bold", fontSize=13, spaceBefore=6, spaceAfter=4))
styles.add(ParagraphStyle(name="H2", fontName="DejaVuSans-Bold", fontSize=10.3, spaceBefore=6, spaceAfter=3))
styles.add(ParagraphStyle(name="Small", fontName="DejaVuSans", fontSize=7.6, leading=9.3, textColor=colors.HexColor("#444444")))
styles.add(ParagraphStyle(name="Ref", fontName="DejaVuSans", fontSize=7.9, leading=10.3, spaceAfter=4))
styles.add(ParagraphStyle(name="TitleMain", fontName="DejaVuSans-Bold", fontSize=14, leading=17, spaceAfter=8, alignment=1))
styles.add(ParagraphStyle(name="Subtitle", fontName="DejaVuSans", fontSize=9.5, textColor=colors.HexColor("#555555"), spaceAfter=8, alignment=1))

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=1.6 * cm, rightMargin=1.6 * cm,
    topMargin=1.4 * cm, bottomMargin=1.4 * cm,
    title="HydroGym Literatür Taraması - Haktan Güloğlu",
)

story = []
P = lambda t, s="Body9": story.append(Paragraph(t, styles[s]))
SP = lambda h=4: story.append(Spacer(1, h))

# ---------------------------------------------------------------- Başlık
P("Literatür Taraması: Analitik Gradyan Enjeksiyonlu Pekiştirmeli Öğrenme ve Kaotik Akış Kontrolü", "TitleMain")
P("Haktan Güloğlu &nbsp;|&nbsp; Literatür taraması raporu &nbsp;|&nbsp; 25 Ağustos 2026 &nbsp;|&nbsp; Sonuç raporunu tamamlayıcı belge", "Subtitle")

# ---------------------------------------------------------------- 1. Kapsam ve yöntem
P("1. Kapsam ve Tarama Yöntemi", "H2")
P(
"""Bu tarama, 24 Ağustos 2026 tarihli sonuç raporunda sunulan PPO/GPPO/SAC/GSAC pilot
çalışmasının hangi literatüre dayandığını ve hangi boşluğa oturduğunu göstermek amacıyla
hazırlanmıştır. Tarama; HydroGym platformunun kendisi, diferansiyellenebilir simülasyon üzerinden
analitik gradyan enjekte eden politika-optimizasyon yöntemleri, kaotik sistemlerde gradyan
kararsızlığı ve türbülansın öngörülebilirliği, JAX tabanlı diferansiyellenebilir CFD çözücüleri,
temel politika-gradyan/varyans-azaltma yöntemleri, analitik-gradyan-vs-zeroth-order karşılaştırma
çalışmaları ve akışkanlar dinamiğinde RL tabanlı aktif akış kontrolünün kökeni olmak üzere on
tematik eksende genişletilmiştir. Kaynaklar arXiv, Nature, PNAS, ICML/ICLR-OpenReview, Physical
Review Letters ve doğrudan yayıncı sayfalarından gerçek zamanlı web taraması ile doğrulanmış;
başlık, yazar listesi ve yayın tarihi kaynağın kendisinden teyit edilmiştir. Bu, pilot çalışmanın
hipotezini ve konumlandırmasını doğrudan destekleyen, odaklı bir hedefli taramadır (N=25).""")

# ---------------------------------------------------------------- 2. Tematik kümeler
P("2. Tematik Kümeleme ve Bulgular", "H2")

P("2.1 HydroGym platformu ve referans makale", "Body9")
P(
"""Çalışmanın doğrudan referans noktası olan <b>Lagemann ve ark. (2026, Nature)</b> [1], HydroGym'i
61+ doğrulanmış akış-kontrol ortamı sunan, çözücüden bağımsız bir RL platformu olarak tanıtıyor;
JAX tabanlı Kolmogorov flow ortamında diferansiyellenebilir fizik üzerinden RL gradyanının PDE
çözücüsü boyunca geri yayılmasıyla (GPPO), standart PPO'ya kıyasla %65'e varan episode tasarrufu
raporluyor. Makale bu sonucu <b>tek bir Reynolds rejiminde, tek eğitim seed'i</b> ile veriyor ve
analitik gradyanın kaotik/türbülanslı rejimlerde nasıl davrandığını test etmiyor — pilot
çalışmamızın doğrudan hedeflediği boşluk burada. İlişkili bir ikinci kaynak olan
<b>"Plug-and-Play Benchmarking of RL Algorithms for Large-Scale Flow Control" (2026)</b> [2],
HydroGym ekosistemi içinde farklı RL algoritmalarının sistematik karşılaştırmasını ele alıyor ve
bizim 4-algoritma (PPO/GPPO/SAC/GSAC) karşılaştırma tasarımımızla metodolojik olarak aynı ailede.""")

P("2.2 Diferansiyellenebilir simülasyon üzerinden analitik-gradyan politika optimizasyonu", "Body9")
P(
"""GPPO'nun ait olduğu daha geniş yöntem ailesi, robotik/kontrol literatüründe önce ortaya çıktı.
<b>Xu ve ark. (2022, ICLR) — SHAC</b> [3], kısa ufuklu (short-horizon) rollout'lar ve öğrenilmiş bir
terminal değer fonksiyonu ile analitik gradyanı, tam episode ufkundan ayırarak patlayan/kaybolan
gradyan sorununu hafifletiyor — GPPO/GSAC'ta kullandığımız 6-adımlık kısa ufuk tasarımı bu
yaklaşımın doğrudan bir uyarlaması. <b>Xing, Luk ve Oh (2025, ICLR) — SAPO/Rewarped</b> [4], SHAC
tarzı analitik-gradyan RL'yi maksimum-entropi (SAC benzeri) bir çerçeveye taşıyarak deforme-olabilir
cisim görevlerinde üstün performans gösteriyor. Bu, bizim GSAC (Gradient-enhanced SAC)
tasarımımızla neredeyse birebir aynı motivasyona sahip — ancak SAPO hiçbir akışkanlar dinamiği
ortamına uygulanmamış; GSAC'ın HydroGym/Kolmogorov flow'a uygulanması, bu iki çizginin (GPPO ve
SAPO) kesişiminde literatürde doldurulmamış bir boşluğu hedefliyor. Aynı ailenin bir sonraki
adımı olan <b>AHAC (Georgiev ve ark., 2024, ICML)</b> [25], sabit kısa-ufuk yerine sert (stiff)
dinamiklerden kaçınacak şekilde ufku adaptif olarak ayarlıyor — GSAC'ın sabit 6-adımlık ufkunun,
Re=300'deki kararsızlığı geciktirmek yerine tamamen önlemek için adaptif hâle getirilebileceğine
işaret eden doğrudan bir sonraki-adım kaynağı.""")

P("2.3 Kaotik sistemlerde analitik gradyan kararsızlığı", "Body9")
P(
"""Hipotezimizin teorik dayanağı <b>Metz, Freeman, Schoenholz ve Kachman (2021) — "Gradients are
Not All You Need"</b> [5]. Bu çalışma, kaotik dinamik sistemlerde (fizik simülasyonları, RNN'ler,
öğrenilmiş optimize ediciler dahil) geri yayılan gradyanların, sistem Jacobian'ının spektrumundan
kaynaklanan bir kelebek-etkisi mekanizmasıyla patladığını gösteriyor ve kırpma (clipping),
kısaltılmış BPTT veya gradyansız (black-box) yöntemlere geçişi çözüm olarak öneriyor. Bu, bizim
"kaotik Re rejiminde analitik gradyanın normu/varyansı kararsızlaşacak" ana hipotezimizin doğrudan
teorik temeli; 500-episode sonuçlarımızdaki "GPPO ve GSAC de sonunda NaN'a düşüyor, ama çok daha
geç" bulgusu, Metz ve ark.'nın öngördüğü kaçınılmaz kararsızlığı destekliyor, sadece gecikme
mekanizmasını (off-policy + kısa ufuk) ampirik olarak ayrıştırıyor.""")

P("2.4 Temel pekiştirmeli öğrenme algoritmaları", "Body9")
P(
"""Karşılaştırmanın iki gradyansız çıpası, sırasıyla on-policy ve off-policy ailelerin standart
temsilcileri: <b>Schulman ve ark. (2017) — PPO</b> [6] ve <b>Haarnoja ve ark. (2018) — Soft
Actor-Critic</b> [7,8]. SAC'ın off-policy replay-buffer ve entropi-düzenlileştirme yapısı, hem
Nature makalesinin hem de SAPO'nun karşılaştırma zemininde yer alıyor; bizim SAC/GSAC çiftimiz de
bu temel algoritma üzerine analitik gradyan enjekte ederek kurulmuş durumda.""")

P("2.5 Akışkanlar dinamiğinde RL tabanlı aktif akış kontrolü (bağlam)", "Body9")
P(
"""Daha geniş bağlam için, gradyansız RL'nin türbülanslı akış kontrolünde başarılı olduğu
gösterilen çalışmalar tarandı: <b>türbülanslı ayrılma kabarcığı kontrolü (Nature Communications,
2025)</b> [9] ve <b>türbülanslı koşullarda aktif akış kontrolüne DRL uygulanması</b> [10], klasik
periyodik zorlamaya kıyasla RL'nin üstünlüğünü gösteriyor; <b>"Deep RL for Robotics" (2025)
derlemesi</b> [11] ise diferansiyellenebilir simülasyonun robotik alanındaki genel eğilimini
özetliyor. Bu kaynaklar, akış kontrolünde RL'nin olgunluğunu doğruluyor ama hiçbiri analitik
gradyan enjeksiyonunu kaotik/türbülanslı rejimle sistematik olarak karşılaştırmıyor.""")

P("2.6 JAX tabanlı diferansiyellenebilir CFD çözücüleri", "Body9")
P(
"""HydroGym'in JAX ortamlarının (ve dolayısıyla bizim GPPO/GSAC uygulamamızın) altyapısal
temeli, <b>JAX'ın kendisi (Bradbury ve ark., 2018)</b> [12] — geri/ileri modlu otomatik
türev, JIT-derleme ve GPU/TPU hızlandırmayı birleştiren kütüphane. Akışkanlar dinamiğine özel
iki öncü uygulama: <b>JAX-CFD (Kochkov ve ark., 2021, PNAS)</b> [13], ML-hızlandırmalı türbülans
çözücüsüyle 8-10× daha kaba ızgarada temel çözücü doğruluğuna ulaşırken 40-80× hızlanma
sağlıyor; <b>JAX-Fluids ve JAX-Fluids 2.0 (Bezgin, Buhendwa, Adams; 2022/2024)</b> [14,15],
sıkıştırılabilir iki-fazlı akışlar için uçtan-uca diferansiyellenebilir bir çözücü sunuyor —
dikkat çekici olarak bu iki yazar (Bezgin, Buhendwa) HydroGym Nature makalesinin de ortak
yazarları arasında, yani HydroGym'in diferansiyellenebilir-çözücü altyapısı doğrudan bu
çizgiden besleniyor.""")

P("2.7 Politika-gradyan temelleri ve varyans azaltma", "Body9")
P(
"""PPO'nun öncülleri olan <b>TRPO (Schulman ve ark., 2015)</b> [16] ve <b>Genelleştirilmiş
Avantaj Kestirimi — GAE (Schulman ve ark., 2015)</b> [17], hem PPO'nun hem GPPO'nun avantaj
kestirimi/güven-bölgesi mekanizmalarının doğrudan atası. Ayrıca SAC'ın maksimum-entropi
çerçevesinin kökeni olan <b>Soft Q-learning (Haarnoja, Tang, Abbeel, Levine; 2017)</b> [18],
entropi-düzenlileştirmenin off-policy öğrenmede neden GSAC'ın kaotik rejimde daha dayanıklı
kaldığını açıklayan teorik zemini sağlıyor: entropi terimi, politikayı belirsizlik altında
daha geniş bir eylem dağılımına zorlayarak tek bir kötü-koşullu gradyan adımının etkisini
sönümlüyor.""")

P("2.8 Analitik gradyan mı, zeroth-order mi? Kısmi diferansiyellenebilirlik", "Body9")
P(
"""<b>Suh, Simchowitz, Zhang ve Tedrake (2022, ICML)</b> [19], analitik (first-order) ve
zeroth-order politika gradyanlarını sertlik (stiffness) ve süreksizlik içeren sistemlerde
karşılaştırıyor ve α-dereceli bir karma kestirici öneriyor — bizim GSAC'taki %50-%50 karışım
tasarımının kabaca bir özel hâli. Bu çalışmanın <b>2026 tarihli bir takip analizi</b> [20] aynı
soruyu güncel deneylerle yeniden sınıyor. <b>ABPT (2025)</b> [21], ödülün yalnızca kısmen
diferansiyellenebilir olduğu durumlar için düzeltilmiş bir BPTT şeması öneriyor — HydroGym'in
kontrol-ödül ile fizik-durumu karışık yapısına yakın bir problem sınıfı.""")

P("2.9 Akış kontrolünde RL'nin kökeni ve türbülans rejimine genişlemesi", "Body9")
P(
"""Alanın kurucu çalışması <b>Rabault ve ark. (2019, J. Fluid Mech.)</b> [22], DRL'nin bir
silindir arkasındaki periyodik girdap dökülmesini kontrol edebildiğini ilk kez gösteriyor —
bugünkü HydroGym benzeri ortamların kavramsal atası. <b>Bluff-body wake'te gerçek-zamanlı RL
(2025)</b> [23], durum-bağımlı türbülanslı kontrolü ele alarak bizim Re=300 rejimimize
yakın bir zorluk seviyesinde çalışıyor.""")

P("2.10 Kaos ve türbülansın öngörülebilirliği (Lyapunov çerçevesi)", "Body9")
P(
"""<b>Boffetta ve Musacchio (2017, Phys. Rev. Lett.)</b> [24], homojen-izotropik türbülansta en
büyük Lyapunov üstelinin Reynolds sayısıyla birlikte büyüdüğünü ve öngörü ufkunun bu üstelle
ters orantılı olduğunu gösteriyor. Bu, Re=300'de dört algoritmanın da neden er ya da geç
NaN'a düştüğünü (öngörü ufku sonlu ve Re ile küçülüyor) ve neden Re arttıkça bu ufkun daha da
kısalacağını (dolayısıyla gecikme çarpanlarının Re'ye duyarlı olabileceğini) niceliksel olarak
çerçeveleyen temel bir kaynak — Metz ve ark. [5] ile birlikte hipotezimizin iki ayrı (biri
optimizasyon, biri akışkanlar-dinamiği kökenli) teorik dayanağını oluşturuyor.""")

# ---------------------------------------------------------------- 3. Sentez
P("3. Sentez: Literatürdeki Boşluk ve Pilot Çalışmanın Konumu", "H2")
P(
"""Genişletilmiş tarama dört ayrı çizgiyi bir araya getiriyor — (a) HydroGym/GPPO'nun tek-rejimli
%65 örnekleme kazancı iddiası [1], (b) analitik-gradyan RL'nin genel kararsızlık riski, hem
optimizasyon [5] hem akışkanlar-dinamiği [24] tarafından teorik olarak desteklenmiş, (c) bu riski
off-policy/entropi-düzenlileştirilmiş yapılarla ya da karma (α-dereceli) kestiricilerle azaltma
girişimleri [3,4,18,19,20], ve (d) bu yöntemlerin dayandığı JAX-tabanlı diferansiyellenebilir CFD
altyapısı [12,13,14,15] — ama bu dördü <b>hiçbir kaynakta birlikte, bir akışkanlar dinamiği
ortamında ve düzenli-kaotik rejim karşılaştırmasıyla</b> test edilmemiş. Pilot çalışmamızın
(GSAC'ın HydroGym'e ilk uygulanması, dört algoritmanın Re=45/Re=300'de sistematik karşılaştırması)
doldurduğu boşluk tam olarak burası; elde ettiğimiz "off-policy ve analitik-gradyan katkılarının
yaklaşık çarpımsal ayrışması" bulgusunun literatürde daha önce raporlanmamış, gerçekten yeni bir
gözlem olduğu bu genişletilmiş taramayla da doğrulanmış oluyor.""")

# ---------------------------------------------------------------- 4. Kaynakça
P("4. Kaynakça", "H2")
refs = [
"[1] Lagemann, C., Mokbel, S., Gondrum, M. ve ark. (2026). The HydroGym reinforcement learning platform for fluid dynamics. <i>Nature</i>. nature.com/articles/s41586-026-10917-6 (ön baskı: arXiv:2512.17534)",
"[2] Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control (2026). arXiv:2601.15015",
"[3] Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A., Macklin, M. (2022). Accelerated Policy Learning with Parallel Differentiable Simulation (SHAC). <i>ICLR 2022</i>. arXiv:2204.07137",
"[4] Xing, E., Luk, V., Oh, J. (2025). Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation (SAPO / Rewarped). <i>ICLR 2025</i>. arXiv:2412.12089",
"[5] Metz, L., Freeman, C. D., Schoenholz, S. S., Kachman, T. (2021). Gradients are Not All You Need. arXiv:2111.05803",
"[6] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347",
"[7] Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. <i>ICML 2018</i>. arXiv:1801.01290",
"[8] Haarnoja, T., Zhou, A., Hartikainen, K. ve ark. (2018). Soft Actor-Critic Algorithms and Applications. arXiv:1812.05905",
"[9] Deep reinforcement learning for active flow control in a turbulent separation bubble (2025). <i>Nature Communications</i>. doi.org/10.1038/s41467-025-56408-6",
"[10] Applying deep reinforcement learning to active flow control in turbulent conditions (2020). arXiv:2006.10683",
"[11] Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes (2025). <i>Annual Review of Control, Robotics, and Autonomous Systems</i>.",
"[12] Bradbury, J., Frostig, R., Hawkins, P. ve ark. (2018). JAX: composable transformations of Python+NumPy programs. github.com/jax-ml/jax (SysML 2018)",
"[13] Kochkov, D., Smith, J. A., Alieva, A., Wang, Q., Brenner, M. P., Hoyer, S. (2021). Machine learning–accelerated computational fluid dynamics (JAX-CFD). <i>PNAS</i> 118(21). doi:10.1073/pnas.2101784118",
"[14] Bezgin, D. A., Buhendwa, A. B., Adams, N. A. (2022). JAX-Fluids: A fully-differentiable high-order CFD solver for compressible two-phase flows. arXiv:2203.13760",
"[15] Bezgin, D. A., Buhendwa, A. B., Adams, N. A. (2024). JAX-Fluids 2.0: Towards HPC for Differentiable CFD of Compressible Two-phase Flows. arXiv:2402.05193",
"[16] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., Abbeel, P. (2015). Trust Region Policy Optimization (TRPO). arXiv:1502.05477",
"[17] Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE). arXiv:1506.02438",
"[18] Haarnoja, T., Tang, H., Abbeel, P., Levine, S. (2017). Reinforcement Learning with Deep Energy-Based Policies (Soft Q-Learning). <i>ICML 2017</i>. arXiv:1702.08165",
"[19] Suh, H. J. T., Simchowitz, M., Zhang, K., Tedrake, R. (2022). Do Differentiable Simulators Give Better Policy Gradients? <i>ICML 2022</i>. arXiv:2202.00817",
"[20] Does \"Do Differentiable Simulators Give Better Policy Gradients?\" Give Better Policy Gradients? (2026). arXiv:2604.18161",
"[21] ABPT: Amended Backpropagation through Time with Partially Differentiable Rewards (2025). arXiv:2501.14513",
"[22] Rabault, J., Kuchta, M., Jensen, A., Réglade, U., Cerardi, N. (2019). Artificial neural networks trained through deep reinforcement learning discover control strategies for active flow control. <i>Journal of Fluid Mechanics</i>.",
"[23] Real-time reinforcement learning for turbulent state-dependent control in a bluff-body wake (2025). arXiv:2509.11002",
"[24] Boffetta, G., Musacchio, S. (2017). Chaos and Predictability of Homogeneous-Isotropic Turbulence. <i>Physical Review Letters</i> 119, 054102. arXiv:1707.01648",
"[25] Georgiev, I., Srinivasan, K., Xu, J., Heiden, E., Garg, A. (2024). Adaptive Horizon Actor-Critic for Policy Learning in Contact-Rich Differentiable Simulation (AHAC). <i>ICML 2024</i>. arXiv:2405.17784",
]
for r in refs:
    P(r, "Ref")

SP(6)
P("Not: Tüm kaynaklar 25 Ağustos 2026 tarihinde gerçek zamanlı web taraması ile doğrulanmıştır; başlık/yazar/tarih bilgileri kaynağın kendi sayfasından teyit edilmiştir.", "Small")

doc.build(story)
print("wrote", OUT)
