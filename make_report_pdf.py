#!/usr/bin/env python3
"""3 sayfayi gecmeyen ozet rapor: hydrogym makale fikri sonuclari + makalede sunum plani (500-ep, 4 algoritma).
Turkce karakterler icin DejaVu Sans (matplotlib ile gelen TTF) reportlab'a gomulu font olarak kaydedilir."""
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
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

OUT = "HydroGym_Sonuc_Raporu_HaktanGuloglu.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body9", fontName="DejaVuSans", fontSize=9, leading=11.5, alignment=TA_JUSTIFY, spaceAfter=4))
styles.add(ParagraphStyle(name="H1", fontName="DejaVuSans-Bold", fontSize=13, spaceBefore=6, spaceAfter=4))
styles.add(ParagraphStyle(name="H2", fontName="DejaVuSans-Bold", fontSize=10.5, spaceBefore=6, spaceAfter=3))
styles.add(ParagraphStyle(name="Small", fontName="DejaVuSans", fontSize=7.7, leading=9.5, textColor=colors.HexColor("#444444")))
styles.add(ParagraphStyle(name="TitleMain", fontName="DejaVuSans-Bold", fontSize=14, leading=17, spaceAfter=8, alignment=1))
styles.add(ParagraphStyle(name="Subtitle", fontName="DejaVuSans", fontSize=9.5, textColor=colors.HexColor("#555555"), spaceAfter=8, alignment=1))

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=1.6 * cm, rightMargin=1.6 * cm,
    topMargin=1.4 * cm, bottomMargin=1.4 * cm,
    title="HydroGym Sonuç Raporu - Haktan Güloğlu",
)

story = []
P = lambda t, s="Body9": story.append(Paragraph(t, styles[s]))
SP = lambda h=4: story.append(Spacer(1, h))

# ---------------- Title ----------------
P("Analitik Gradyanın Akış Kaotikliğine Duyarlılığı: HydroGym'de PPO / GPPO / SAC / GSAC Karşılaştırması", "TitleMain")
P("Haktan Güloğlu &nbsp;|&nbsp; Sonuç raporu &nbsp;|&nbsp; 24 Ağustos 2026 &nbsp;|&nbsp; 4 algoritma, 500 episode, n=1 seed", "Subtitle")

# ---------------- 1. Motivasyon + Yöntem ----------------
P("1. Motivasyon ve Yöntem", "H1")
P(
    "HydroGym'i tanıtan Nature makalesi (Lagemann ve ark., 2026), diferansiyellenebilir akış simülatörleri "
    "üzerinde analitik gradyan enjeksiyonunun (GPPO) Kolmogorov flow ortamında standart PPO'ya kıyasla "
    "<b>%65 daha az eğitim episode'u</b> ile aynı nihai ödüle ulaştığını gösteriyor (Şekil 3a-b, tek seed). "
    "Ancak bu sonuç <b>tek bir Reynolds rejiminde</b> raporlanmış; analitik gradyanın düzenli (laminer) ve "
    "kaotik (türbülanslı) rejimlerde farklı davranıp davranmadığı hiç test edilmemiş. Diferansiyellenebilir "
    "simülasyon literatüründe bilinen bir risk, kaotik sistemlerde backprop-through-time zincirinin kelebek "
    "etkisiyle kararsızlaşması (gradyan normunun patlaması/anlamsızlaşması).",
    "Body9",
)
P(
    "<b>Hipotez:</b> analitik gradyanın normu/varyansı düşük Re'de kararlı, yüksek Re'de kararsız olacak; "
    "buna karşı off-policy SAC ailesi (bizim eklememiz: <b>GSAC = SAC + analitik gradyan</b>, literatürde "
    "HydroGym'e hiç uygulanmamış bir kombinasyon), on-policy GPPO'dan daha dayanıklı kalacak.",
    "Body9",
)
P(
    "<b>Yöntem:</b> Gerçek <font face='Courier'>hydrogym.jax.envs.kolmogorov.KolmogorovFlow</font> ortamı "
    "(64x64 grid, pseudo-spektral IMEX-RK4CN çözücü) üzerinde <b>4 algoritma (PPO, GPPO, SAC, GSAC), 2 Re "
    "rejiminde, her biri 500 episode/n=1 seed ile</b> eğitildi: <b>Re=45</b> (düzenli) ve <b>Re=300</b> "
    "(kaotik - kütüphanenin bu grid/dt'deki kendi kararlılık sınırı ~Re 350-380 olduğu için orijinal hedef "
    "Re=450 yerine seçildi). GPPO/GSAC'ta analitik gradyan, 6 adımlık diferansiyellenebilir rollout'un "
    "jax.grad'i olarak hesaplanıp standart gradyanla %50-%50 karıştırıldı. Tüm 8 config (4 algo x 2 Re), "
    "GPU'nun tek seferde tek JAX süreci kaldırabilmesi nedeniyle sıralı çalıştırıldı (toplam ~3.3 saat).",
    "Body9",
)

# ---------------- 2. Sonuçlar ----------------
P("2. Elde Edilen Sonuçlar", "H1")

P("2.1 Re=45 (düzenli akış): örneklem verimliliği ve nihai performans", "H2")
P(
    "<b>GPPO vs PPO, makaleyle doğrudan karşılaştırma:</b> GPPO, PPO'nun 500 episode'da ulaştığı en iyi "
    "(smoothed) ödül seviyesine <b>episode 162'de</b> ulaşırken PPO aynı seviyeye ancak <b>episode 438'de</b> "
    "ulaşıyor - <b>%63 episode farkı</b>, makalenin raporladığı %65'e çok yakın. Bu, implementasyonumuzun "
    "makaledeki mekanizmayı doğru yakaladığının güçlü bir işareti. <b>SAC vs GSAC:</b> off-policy yapı zaten "
    "çok daha verimli olduğundan (SAC/GSAC son-5-episode ortalama ödülü -16.7/-18.0, PPO/GPPO -25.1/-25.4 "
    "civarında kalıyor - bkz. Şekil A sol panel), GSAC bu ayarlarda SAC'in nihai performansını geçemedi; "
    "analitik gradyanın PPO ailesindeki örneklem-hızlandırma etkisi SAC ailesinde aynı biçimde gözlenmiyor.",
    "Body9",
)

P("2.2 Re=300 (kaotik akış): dördü de çöküyor, ama ÇOK farklı sürelerde - ana bulgu", "H2")
tbl_data = [["Algoritma", "İlk NaN episode'u", "PPO'ya göre gecikme"],
            ["PPO", "12", "1.0x"],
            ["SAC", "36", "3.0x"],
            ["GPPO", "83", "6.9x"],
            ["GSAC", "141", "11.8x"]]
t = Table(tbl_data, colWidths=[4.0 * cm, 4.5 * cm, 4.5 * cm])
t.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
    ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
]))
story.append(t)
SP(4)
P(
    "Makalenin hiç test etmediği kaotik rejimde <b>dört algoritmanın da 500 episode'u kararlı tamamlayamadığı</b>, "
    "hepsinin er ya da geç NaN'a düştüğü görüldü - ama çöküşe kadar geçen süre <b>11.8 kata kadar</b> "
    "farklılaştı. Desen neredeyse çarpımsal: off-policy yapı tek başına <b>3.0x</b> gecikme sağlıyor "
    "(SAC/PPO); analitik gradyan tek başına on-policy zeminde <b>6.9x</b> (GPPO/PPO), off-policy zeminde "
    "<b>3.9x</b> (GSAC/SAC) gecikme sağlıyor. İkisi birlikte (GSAC) <b>11.8x</b>'e ulaşıyor - 3.0 x 3.9 ≈ "
    "11.7 ile çarpıcı derecede tutarlı (n=1 ile bu kesin bir yasa değil, ama desen net). <b>Dürüst yorum:</b> "
    "analitik gradyan kaotik rejimde kararsızlığı azaltmıyor, <i>erteliyor</i>; GSAC'ın off-policy + "
    "analitik-gradyan kombinasyonu, tek başına her iki mekanizmadan da daha dayanıklı.",
    "Body9",
)

img = Image("all4_500ep_comparison.png", width=17.4 * cm, height=6.96 * cm)
story.append(img)
P("Şekil A. 500 episode, 4 algoritma — sol: Re=45 (ödül eğrileri), sağ: Re=300 (ilk NaN episode'ları işaretli).", "Small")
SP(6)

img3 = Image("stability_delay_500ep.png", width=8.5 * cm, height=6.07 * cm)
story.append(img3)
P("Şekil B. Re=300'de ilk NaN episode'u — algoritmaya göre kararsızlığa düşme süresi (PPO'ya göre kaç kat).", "Small")

P("3. Bu Sonuçların Makalede Nasıl Sunulacağı", "H1")
P(
    "<b>Çerçeve:</b> Nature makalesinin Şekil 3a-b sonucunu iki eksende genişleten kısa bir teknik not / "
    "pilot bulgu olarak sunulacak - makalenin ana iddiasını (GPPO faydalıdır) çürütmüyoruz, onu <i>hangi "
    "koşullarda ve neden</i> geçerli olduğu sorusuna genişletiyoruz.",
    "Body9",
)
P(
    "<b>(1) Rejim genişlemesi:</b> Şekil A sol panel makaledeki Şekil 3b'nin doğrudan karşılaştırması olarak "
    "kullanılacak (reprodüksiyon doğrulaması, %63 vs %65). Sağ panel + Şekil B, makalenin hiç göstermediği "
    "kaotik rejimdeki ana bulguyu taşıyacak: <i>GPPO'nun avantajı Re=300'de nitelik değiştiriyor - 'daha az "
    "episode'da daha iyi ödül' yerine 'daha uzun süre kararlı kalabilme', ve bu avantaj sonunda tükeniyor.</i>",
    "Body9",
)
P(
    "<b>(2) Mekanizma ayrıştırması (GSAC katkısı, makalenin en genişletilebilir noktası):</b> Şekil B'deki "
    "çarpımsal desen (off-policy x analitik-gradyan ≈ toplam gecikme), kararlılık avantajının tek boyutlu "
    "'analitik gradyan iyidir' anlatısından, 'iki mekanizma bağımsız ve yaklaşık çarpımsal katkı sağlıyor' "
    "şeklinde daha zengin, test edilebilir bir hipoteze taşınmasını öneren somut bir sonuç olarak sunulacak. "
    "Bu, literatürde HydroGym'e hiç uygulanmamış GSAC kombinasyonunun ilk defa sağladığı bir gözlem.",
    "Body9",
)
P(
    "<b>Özet mesaj (makale ile aradaki fark):</b> Düzenli rejimde "
    "(Re=45) sonucumuz makaleyle pratikte aynı (%63 vs %65). Kaotik rejimde (Re=300, makalede hiç test "
    "edilmemiş) analitik gradyan ve off-policy yapı, birbirinden bağımsız ve yaklaşık çarpımsal şekilde "
    "kararsızlığı erteliyor - en dayanıklı kombinasyon (GSAC) PPO'ya göre 11.8 kat, ama hiçbiri kalıcı "
    "kararlılık sağlamıyor.",
    "Body9",
)

P("4. Sınırlamalar (dürüstçe)", "H1")
lim_items = [
    "<b>n=1 seed</b> her yerde - hiçbir sonuç istatistiksel olarak sağlam değil; \"trend var\" diyebiliriz, "
    "\"kanıtlanmıştır\" diyemeyiz. Makalenin Şekil 3b'si de tek seed, ama ana sonuçları (Şekil 2, Extended "
    "Data) n=3 kullanıyor. Artık tüm 4 algoritma aynı 500-episode bütçesinde (önceki sürümde SAC/GSAC 25 "
    "episode'da kalmıştı).",
    "11.8x / 6.9x / 3.9x / 3.0x gecikme çarpanları, n=1 seed ile \"sistematik bir yasa\" olarak "
    "genellenemez - tek bir rastgele başlangıç/aksiyon dizisinin sonucu olabilir. Çarpımsal tutarlılık "
    "(3.0x3.9≈11.7≈11.8) ilgi çekici ama tesadüfi olabilir; çoklu seed'le doğrulanmadan bir nokta-tahmin "
    "gözlemi olarak okunmalı.",
    "Re=300, orijinal hedef Re=450 yerine kullanıldı - kütüphanenin çözücüsü bu grid/dt'de ~Re 350-380 "
    "civarında kararlılık sınırına sahip.",
    "Kısa episode'lar (15 RL-adımı) ve kısa analitik ufuk (6 adım) - hem RL yakınsaması hem Lyapunov "
    "büyümesinin tam gözlemlenmesi için yetersiz olabilir.",
    "GSAC karışım ağırlığı sabit %50-%50 - adaptif ağırlıklandırma denenmedi.",
]
for li in lim_items:
    P("&bull; " + li, "Body9")

P(
    "<b>Sonraki adım:</b> Aynı deneyi 3-5 farklı seed ile tekrarlayıp gecikme çarpanlarının tutarlılığını "
    "ve çarpımsallık gözleminin rastlantı olup olmadığını istatistiksel olarak test etmek.",
    "Body9",
)

doc.build(story)
print("wrote", OUT)
