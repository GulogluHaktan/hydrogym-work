#!/usr/bin/env python3
"""Full bildiri PDF (n=2, seed=0+seed=1 findings), all tables/figures embedded.
Turkce karakterler icin DejaVu Sans (matplotlib ile gelen TTF) reportlab'a
gomulu font olarak kaydedilir. Kaynak icerik: BILDIRI_TASLAK.md."""
import re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    KeepTogether, PageBreak, HRFlowable
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
pdfmetrics.registerFont(TTFont("DejaVuSansMono", MPL_FONT_DIR + "DejaVuSansMono.ttf"))

OUT = "reports/Kaos_Sininda_Gradyan_HaktanGuloglu.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body9", fontName="DejaVuSans", fontSize=9.3, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5))
styles.add(ParagraphStyle(name="BodyItalic", fontName="DejaVuSans-Oblique", fontSize=9.3, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5))
styles.add(ParagraphStyle(name="H1", fontName="DejaVuSans-Bold", fontSize=13.5, leading=16, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1F4E78")))
styles.add(ParagraphStyle(name="H2", fontName="DejaVuSans-Bold", fontSize=10.8, leading=13, spaceBefore=9, spaceAfter=4, textColor=colors.HexColor("#2c3e50")))
styles.add(ParagraphStyle(name="H3", fontName="DejaVuSans-Bold", fontSize=9.6, leading=12, spaceBefore=6, spaceAfter=3))
styles.add(ParagraphStyle(name="Small", fontName="DejaVuSans", fontSize=7.6, leading=9.6, textColor=colors.HexColor("#444444"), spaceAfter=4))
styles.add(ParagraphStyle(name="Caption", fontName="DejaVuSans-Oblique", fontSize=8.2, leading=10.5, textColor=colors.HexColor("#333333"), spaceAfter=10, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="TitleMain", fontName="DejaVuSans-Bold", fontSize=16.5, leading=20, spaceAfter=4, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="Subtitle", fontName="DejaVuSans-Oblique", fontSize=10.5, textColor=colors.HexColor("#555555"), spaceAfter=10, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="AuthorLine", fontName="DejaVuSans", fontSize=9.5, textColor=colors.HexColor("#333333"), spaceAfter=2, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="StatusBox", fontName="DejaVuSans", fontSize=8.3, leading=11, textColor=colors.HexColor("#333333"), spaceAfter=4, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name="Kw", fontName="DejaVuSans", fontSize=8.6, leading=11, spaceAfter=4, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name="TblHead", fontName="DejaVuSans-Bold", fontSize=7.6, leading=9.2, textColor=colors.white, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TblCell", fontName="DejaVuSans", fontSize=7.6, leading=9.4, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TblCellL", fontName="DejaVuSans", fontSize=7.6, leading=9.4, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="RefBody", fontName="DejaVuSans", fontSize=8.3, leading=10.6, alignment=TA_JUSTIFY, spaceAfter=6, leftIndent=14, firstLineIndent=-14))

PAGE_W, _ = A4
MARGIN = 1.6 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=1.3 * cm, bottomMargin=1.4 * cm,
    title="Kaos Sinirinda Gradyan - Haktan Guloglu",
)

story = []


def md(text):
    """Convert a light markdown subset (**bold**, *italic*, `code`) to reportlab mini-html."""
    text = text.replace("&", "&amp;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="DejaVuSansMono" size="8.4">\1</font>', text)
    return text


def P(text, style="Body9"):
    story.append(Paragraph(md(text), styles[style]))


def SP(h=5):
    story.append(Spacer(1, h))


def H1(n, text):
    story.append(Paragraph(f"{n}. {md(text)}" if n else md(text), styles["H1"]))


def H2(text):
    story.append(Paragraph(md(text), styles["H2"]))


def bullet(text):
    P("• " + text, "Body9")


def hr():
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#bbbbbb"), spaceBefore=6, spaceAfter=8))


def fig(path, caption, width_cm=17.6, height_cm=None, ratio=0.4):
    w = width_cm * cm
    h = (width_cm * ratio) * cm if height_cm is None else height_cm * cm
    story.append(Image(path, width=w, height=h))
    P(caption, "Caption")
    SP(4)


def cellP(text, head=False):
    return Paragraph(md(text), styles["TblHead" if head else "TblCell"])


def cellPL(text, head=False):
    return Paragraph(md(text), styles["TblHead" if head else "TblCellL"])


def mktable(header, rows, col_widths, first_col_left=True, caption=None):
    data = [[cellP(h, head=True) for h in header]]
    for r in rows:
        row = []
        for i, v in enumerate(r):
            if i == 0 and first_col_left:
                row.append(cellPL(v))
            else:
                row.append(cellP(v))
        data.append(row)
    t = Table(data, colWidths=[w * cm for w in col_widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    if caption:
        SP(3)
        P(caption, "Caption")
    SP(6)


# ============================================================ TITLE PAGE ====
P("Kaos Sınırında Gradyan:<br/>HydroGym'de Analitik Gradyan Yöntemlerinin Kırılma Noktası", "TitleMain")
P("(Subtitle: HydroGym Kolmogorov Akışında PPO, GPPO, SAC ve GSAC'ın Kırılma-Noktası Karşılaştırması)", "Subtitle")
P("Haktan Güloğlu", "AuthorLine")
P("Sakarya Üniversitesi, Milli Teknoloji Atölyesi, Yapay Zeka Merkezi", "AuthorLine")
P("haktanguloglu24@gmail.com", "AuthorLine")
SP(8)

status_box_data = [[Paragraph(md(
    "<b>Durum:</b> Genişletilmiş taslak, tam kaynakçalı (n=25 literatür taraması entegre edilmiştir). "
    "İkinci bir eğitim seed'i (seed=1) ile tam 8-konfigürasyonluk doğrulama koşusu <b>tamamlanmıştır</b>; "
    "bulgular n=2 (seed=0, seed=1) üzerinden raporlanmaktadır. Bu doğrulama, n=1'de gözlenen \"temiz "
    "çarpımsal\" gecikme deseninin ikinci seed'te <b>tekrarlanmadığını</b> — özellikle off-policy yapının "
    "tek başına kararlılığı artırdığı iddiasının yön değiştirdiğini — ortaya çıkarmıştır; buna karşın "
    "<b>GSAC'ın her iki seed'de de dört yöntem içinde en dayanıklısı olması tutarlı kalmıştır</b>."
    ), styles["StatusBox"])]]
sbt = Table(status_box_data, colWidths=[CONTENT_W])
sbt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3f8")),
    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#1F4E78")),
    ("TOPPADDING", (0, 0), (-1, -1), 7),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
]))
story.append(sbt)
SP(10)

# ============================================================ ÖZET =========
H2("Özet")
P(
    "Bu çalışma, analitik (diferansiyellenebilir-simülasyon üzerinden hesaplanan) politika gradyanı "
    "kullanan pekiştirmeli öğrenme yöntemlerinin **kaos sınırındaki kırılma noktasını** — eğitimin sayısal "
    "olarak ıraksadığı ilk episode'u — kaotik bir akışkanlar dinamiği kontrol görevinde nicel olarak "
    "karakterize eder. *Nature*'da yayımlanan HydroGym platform makalesi [1], 61'den fazla doğrulanmış akış "
    "kontrolü ortamı sunan, çözücüden bağımsız bir pekiştirmeli öğrenme (RL) platformu tanımlar ve JAX "
    "tabanlı diferansiyellenebilir Kolmogorov akışı ortamında **Gradyan-Zenginleştirilmiş PPO (GPPO)**'nun "
    "eğitim iterasyonlarını standart PPO'ya kıyasla **en az %65 azalttığını** raporlar. Ancak bu sonuç, "
    "makalenin tanımladığı geniş Reynolds aralığının (Re=40–500) **yalnızca tek bir noktasında, tek eğitim "
    "seed'iyle** gösterilmiştir; analitik gradyan enjeksiyonunun düzenli (laminer) ve kaotik (türbülanslı) "
    "rejimlerde nasıl farklı davrandığı ayrıca test edilmemiştir. Bu boşluk, kaotik dinamiklerde geri-yayılan "
    "gradyanların sistem Jacobian'ının spektrumu nedeniyle kararsızlaştığını gösteren genel bir bulguyla [7] "
    "ve türbülansın Lyapunov üstelinin Reynolds sayısıyla büyüdüğünü gösteren klasik bir sonuçla [25] "
    "örtüşmektedir."
)
P(
    "Bu çalışma, gerçek `hydrogym.jax.envs.kolmogorov.KolmogorovFlow` ortamı üzerinde dört algoritmayı — "
    "PPO, GPPO, literatürde HydroGym'e daha önce uygulanmamış SAC [8] ve onun analitik-gradyanla "
    "zenginleştirilmiş hâli **GSAC** — düzenli (Re=45) ve kaotik (Re=300) rejimlerde, 500 episode'luk "
    "bütçelerle, **iki bağımsız eğitim seed'i (seed=0, seed=1)** üzerinden test eder. Re=45'te GPPO'nun "
    "örnekleme avantajı her iki seed'de de doğrulanır (seed=0'da %63, seed=1'de %77 episode tasarrufu; "
    "makalenin bildirdiği %65'e [1] her iki değer de yakın büyüklük mertebesindedir). Re=300'de ise dört "
    "algoritma da sayısal kararsızlığa (NaN) düşer, ancak kırılma zamanı hem algoritmaya hem de **seed'e** "
    "göre çarpıcı biçimde farklılaşır: seed=0'da PPO 12., SAC 36., GPPO 83., GSAC 141. episode'da çöker "
    "(PPO'ya göre 3.0×, 6.9×, 11.8× gecikme); seed=1'de ise PPO 35., **SAC 6.** (PPO'dan **daha erken**, "
    "0.17×), GPPO 40., GSAC 63. episode'de çöker. İkinci seed, n=1'de gözlenen \"off-policy yapı tek başına "
    "kararlılığı artırır\" iddiasını **doğrulamamış**, hatta tersine çevirmiştir; buna karşın analitik "
    "gradyan enjeksiyonunun gecikme yönü (GPPO>PPO, GSAC>SAC) ve **GSAC'ın dört yöntem içinde her iki "
    "seed'de de en dayanıklısı olması** tutarlı kalmıştır. Ham eğitim eğrilerinin incelenmesi, kırılmanın "
    "öncülsüz/ani bir olay olduğunu ve kalıcı-çöküş öncesi kısa bir \"hayatta kalma\" penceresinin (yalnızca "
    "seed=0'ın GSAC'ında gözlenen) bir replay-tamponu seyreltme mekanizmasıyla nicel olarak "
    "açıklanabildiğini — ve bu mekanizmanın seed=1'in farklı sonucunu da doğru öngördüğünü — göstermektedir "
    "(§5.3). Bulgular, analitik gradyan enjeksiyonunun kaotik rejimde kararsızlığı ortadan kaldırmadığını, "
    "yalnızca **ertelediğini** ve bu erteleme büyüklüğünün seed'e bağlı olarak büyük varyans gösterdiğini; "
    "off-policy yapının bu erteleyici etkiyi yalnızca analitik gradyanla **birlikte** güvenilir biçimde "
    "sağladığını ortaya koymaktadır."
)
P(
    "<b>Anahtar Kelimeler:</b> pekiştirmeli öğrenme, diferansiyellenebilir simülasyon, analitik politika "
    "gradyanı, akışkanlar dinamiği kontrolü, kaotik dinamikler, PPO, SAC", "Kw"
)

H2("Abstract")
P(
    "*(Title: \"Gradient at the Edge of Chaos: The Breaking Point of Analytic Gradient Methods in "
    "HydroGym\")* This work characterizes the **breaking point at the edge of chaos** — the first episode "
    "at which training numerically diverges — for analytic-gradient reinforcement learning methods on a "
    "chaotic fluid-dynamics control task. The *Nature* paper introducing HydroGym [1] reports that "
    "Gradient-Enhanced PPO (GPPO) reduces training iterations on the differentiable Kolmogorov-flow "
    "environment by at least 65% relative to standard PPO, demonstrated at a single point within the "
    "paper's tested Reynolds range (Re=40–500) with a single training seed. We evaluate PPO, GPPO, SAC [8], "
    "and a novel gradient-enhanced SAC variant (GSAC) on the real `hydrogym.jax` Kolmogorov-flow environment "
    "at a regular (Re=45) and a chaotic (Re=300) operating point, across **two independent training seeds "
    "(seed=0, seed=1)**. At Re=45, GPPO's sample-efficiency advantage is reproduced in both seeds (63% and "
    "77% fewer episodes to reach PPO's best reward). At Re=300, all four algorithms eventually diverge "
    "numerically, but the episode at which divergence first occurs differs sharply by both algorithm *and* "
    "seed: in seed=0, PPO breaks at episode 12, SAC at 36, GPPO at 83, GSAC at 141 (3.0×, 6.9×, 11.8× delay "
    "vs. PPO); in seed=1, PPO breaks at 35, **SAC at 6** (*earlier* than PPO, 0.17×), GPPO at 40, GSAC at "
    "63. The second seed does **not** reproduce the n=1 claim that off-policy training alone delays "
    "instability — that direction reverses — but the delaying effect of analytic-gradient injection and, "
    "most importantly, **GSAC being the most robust of the four algorithms**, replicate in both seeds. A "
    "replay-buffer dilution mechanism quantitatively explains a short post-trigger survival window observed "
    "only for GSAC in seed=0, and correctly predicts its *absence* in seed=1 (§5.3). We conclude that "
    "analytic gradient injection delays — but does not eliminate — instability under chaotic dynamics, that "
    "the size of this delay varies substantially across seeds, and that off-policy training only reliably "
    "contributes to this delay when combined with analytic gradient injection.", "BodyItalic"
)
story.append(PageBreak())

# ============================================================ 1. GİRİŞ ======
H1(1, "Giriş")
P(
    "Akışkanlar dinamiğinin aktif kontrolü — sürükleme azaltımı, karışım artırımı, aşırı olay bastırma gibi "
    "hedeflerle — pekiştirmeli öğrenmenin son yıllarda en aktif uygulama alanlarından biri hâline gelmiştir. "
    "Alanın kurucu çalışması olan Rabault ve ark. [23], derin RL'nin bir silindir arkasındaki periyodik "
    "girdap dökülmesini kontrol ederek Re=100'de %8 sürükleme azaltımı sağlayabildiğini ilk kez göstermiş; "
    "bu çizgi, türbülanslı ayrılma kabarcığı kontrolü [10] ve gerçek zamanlı, durum-bağımlı türbülanslı iz "
    "(wake) kontrolü [24] gibi çalışmalarla türbülans rejimine genişlemiştir. HydroGym [1], bu dağınık kod "
    "tabanlarını tek, çözücüden bağımsız bir platformda birleştirerek laminerden türbülansa 61'den fazla "
    "ortam sunar ve LBM, FEM gibi klasik çözücülerin yanı sıra JAX [13] tabanlı, uçtan uca "
    "**diferansiyellenebilir** bir çözücü de içerir — bu çözücü altyapısı, JAX-CFD [14] ve JAX-Fluids [15,16] "
    "gibi öncü diferansiyellenebilir CFD kütüphanelerinin doğrudan bir devamıdır. Benzer motivasyonla "
    "geliştirilen, PyTorch tabanlı ve tamamen diferansiyellenebilir bir başka akış-kontrolü kıyaslama paketi "
    "olan FluidGym [2] da bu ekosistemin metodolojik bir akrabasıdır."
)
P(
    "Diferansiyellenebilirlik, standart model-serbest RL'nin ötesine geçip **analitik politika "
    "gradyanlarını** — ödülün politika parametrelerine göre türevini, örnekleme yerine doğrudan zincir "
    "kuralıyla, simülatörün kendisi üzerinden hesaplama imkânı sunar. HydroGym makalesi bu imkânı GPPO ile "
    "gösterir ve Kolmogorov akışı ortamında eğitim iterasyonlarını en az %65 azalttığını raporlar [1]. Ne var "
    "ki bu sonuç **tek bir çalışma noktasında** gösterilmiş olup, makalenin tanımladığı geniş Re=40–500 "
    "aralığının düzenli ve kaotik uçlarında analitik gradyanın davranışı ayrıca karşılaştırılmamıştır."
)
P(
    "Bu boşluk, diferansiyellenebilir programlama literatüründe iyi bilinen bir riskle örtüşür. Metz ve ark. "
    "[7], yinelenen haritalarda, sayısal fizik simülasyonlarında ve öğrenilmiş optimize edicilerde ortak bir "
    "\"kaos-kaynaklı başarısızlık modu\" göstermiş; bunu sistemin Jacobian'ının spektrumuna bağlamıştır. "
    "Akışkanlar-dinamiği tarafında ise Boffetta ve Musacchio [25], homojen-izotropik türbülansta en büyük "
    "Lyapunov üstelinin Reynolds sayısıyla birlikte büyüdüğünü ve öngörü ufkunun bu üstelle ters orantılı "
    "olarak daraldığını göstermiştir. Bu iki bağımsız teorik dayanak, kaotik bir rejimde analitik gradyanın "
    "normunun/yönünün kelebek etkisiyle aşırı duyarlı hâle gelmesini ve GPPO'nun bildirilen avantajının Re "
    "arttıkça zayıflayabileceğini, hatta tersine dönebileceğini düşündürür."
)
P(
    "Analitik gradyan enjeksiyonu şimdiye dek ağırlıklı olarak **on-policy** yöntemlerle veya "
    "rijit-gövde/temaslı robotik görevlerinde uygulanmıştır: SHAC [3], kısa-ufuklu rollout'lar ve öğrenilmiş "
    "bir terminal değer fonksiyonuyla patlayan/kaybolan gradyan sorununu hafifletir; AHAC [5] bu fikri, "
    "ufku sert dinamiklerden kaçınacak şekilde **adaptif olarak** ayarlayarak bir adım öteye taşır. "
    "Off-policy tarafta ise SAPO [4], analitik-gradyan RL'yi maksimum-entropi bir çerçeveye taşıyarak "
    "deforme-olabilir cisim görevlerinde üstün performans göstermiştir — ancak bu yaklaşım hiç akışkanlar "
    "dinamiğine uygulanmamıştır. Bu çalışma, SAC [8]'ın analitik gradyanla zenginleştirilmiş bir versiyonu "
    "olan **GSAC**'ı dördüncü yöntem olarak ekleyerek, GPPO (SHAC ailesi) ile SAPO ailesinin kesişimini "
    "akışkanlar dinamiğine taşır."
)

H2("Katkılar")
bullet("HydroGym'in Kolmogorov akışı üzerindeki GPPO-vs-PPO sonucunu [1], tek bir Re noktasından düzenli–kaotik bir rejim karşılaştırmasına genişletiyoruz.")
bullet("Literatürde HydroGym'e hiç uygulanmamış bir GSAC (SAC + analitik gradyan, SAPO [4] ailesinin akışkanlar dinamiğine ilk taşınması) varyantı sunuyor ve kararlılık avantajının kaynağının \"analitik gradyan\" mı yoksa \"off-policy eğitim yapısı\" mı olduğunu ilk kez ayrıştırıyoruz.")
bullet(
    "Kaotik rejimde dört algoritmanın da sayısal olarak ıraksadığını, ama ıraksama zamanının hem "
    "algoritmaya hem de eğitim seed'ine göre büyük ölçüde değiştiğini (12–141 episode aralığında) "
    "gösteriyoruz. **İki bağımsız seed'in** karşılaştırılması, n=1'de gözlenen \"temiz çarpımsal katkı\" "
    "deseninin bir yapıt olduğunu ortaya koyuyor: analitik gradyanın geciktirici etkisi yön olarak her iki "
    "seed'de de tutarlı kalırken (GPPO>PPO, GSAC>SAC), off-policy yapının **tek başına** geciktirici "
    "olduğu iddiası seed=1'de tersine dönüyor. Tek istikrarlı bulgu, **GSAC'ın her iki seed'de de en "
    "dayanıklısı olması**dır."
)
bullet(
    "Ham eğitim eğrilerini doğrudan inceleyerek kırılmanın mekanizmasını karakterize ediyoruz: kırılma "
    "öncesinde görünür bir öncül yoktur (ani, stokastik bir tetiklenme, her iki seed'de de gözlenir); "
    "off-policy yöntemlerde replay tamponunun zehirli geçişleri seyreltmesi birkaç episode'luk bir "
    "kalıcı-çöküş ertelemesi yaratabilir — bu mekanizmadan türetilen nicel tahmin, seed=0'daki 2-episode'luk "
    "ertelemeyi **ve** seed=1'de erteleme gözlenmemesini **her ikisini de** doğru öngörür (§5.3)."
)
SP(3)
P(
    "<b>Hipotez:</b> Analitik gradyanın normu/varyansı düşük Re'de kararlı, yüksek Re'de kararsız olacak; bu "
    "kararsızlığa karşı GSAC'ın off-policy yapısı, GPPO'nun on-policy yapısından daha dayanıklı kalacaktır. "
    "<b>(n=2 sonucu:</b> GSAC'ın GPPO'dan daha dayanıklı olması her iki seed'de de doğrulanmıştır (seed=0: "
    "141 vs 83; seed=1: 63 vs 40); ancak bunun nedeni \"off-policy yapı tek başına daha dayanıklı\" değil, "
    "off-policy yapının yalnızca analitik gradyanla <b>birlikte</b> güvenilir bir katkı sağlamasıdır — bkz. "
    "§5.2.)"
)

# ============================================================ 2. İLGİLİ ÇALIŞMALAR ==
H1(2, "İlgili Çalışmalar")
H2("2.1 RL ile Akış Kontrolü ve HydroGym")
P(
    "HydroGym [1], farklı çözücüleri tek bir RL arayüzü altında birleştirerek laminerden türbülansa 61+ "
    "ortam sunar; öğrenilen kontrol politikalarının ortamlar arasında sıfır-atış transfer edilebildiğini "
    "(ör. 3B kanat kesitinde %38 sürükleme azaltımı) ve GPPO ile Kolmogorov akışında ≥%65 örnekleme kazancı "
    "sağlandığını gösterir. FluidGym [2], metodolojik olarak akraba, PyTorch-tabanlı ve tamamen "
    "diferansiyellenebilir bir başka akış-kontrolü kıyaslama paketidir. Bu çalışma, HydroGym'in bildirdiği "
    "kazanımın Reynolds sayısına duyarlılığını inceler."
)
H2("2.2 Diferansiyellenebilir Simülasyon Üzerinden Analitik Politika Gradyanları")
P(
    "GPPO'nun ait olduğu daha geniş yöntem ailesi, önce robotik/kontrol literatüründe ortaya çıkmıştır. SHAC "
    "[3], kısa-ufuklu rollout'lar ve pürüzsüz bir kritik fonksiyonuyla analitik gradyanı tam episode "
    "ufkundan ayırarak patlayan/kaybolan gradyan sorununu hafifletir; muscle-actuated lokomosyon "
    "görevlerinde 17 kattan fazla eğitim-süresi azalması bildirir — GPPO/GSAC'ta kullanılan 6-adımlık kısa "
    "ufuk tasarımı bu yaklaşımın doğrudan bir uyarlamasıdır. AHAC [5], ufku sert dinamiklerden kaçınacak "
    "şekilde adaptif ayarlayarak lokomosyon görevlerinde model-serbest temel çizgilere göre %40 daha fazla "
    "ödül elde eder. SAPO [4], bu fikri maksimum-entropi off-policy bir çerçeveye taşır. Bu çalışmadaki "
    "GSAC, SAPO'nun temel fikrini akışkanlar dinamiğine taşıyan ilk uygulamadır."
)
H2("2.3 Kaotik Sistemlerde Analitik Gradyan Kararsızlığı")
P(
    "Hipotezimizin teorik dayanağı Metz ve ark. [7]: kaotik dinamik sistemlerde geri yayılan gradyanların, "
    "sistem Jacobian'ının spektrumundan kaynaklanan bir kelebek-etkisi mekanizmasıyla patladığını gösterir. "
    "Bu genel gözlemin akışkanlar-dinamiği karşılığı Boffetta ve Musacchio [25]: homojen-izotropik "
    "türbülansta en büyük Lyapunov üstelinin Reynolds sayısıyla büyüdüğünü ve öngörü ufkunun bu üstelle "
    "ters orantılı olarak daraldığını gösterir. Bu iki kaynak birlikte, sonuçlarımızdaki \"GPPO ve GSAC de "
    "sonunda NaN'a düşüyor, ama çok daha geç\" bulgusunun teorik zeminini oluşturur."
)
H2("2.4 Analitik Gradyan mı, Zeroth-Order mi? Kısmi Diferansiyellenebilirlik")
P(
    "Suh ve ark. [20], analitik ve zeroth-order politika gradyanlarını sertlik ve süreksizlik içeren "
    "sistemlerde karşılaştırır ve bir α-dereceli karma kestirici önerir — GSAC'taki %50-%50 karışım "
    "tasarımının kabaca özel bir hâlidir. Bu çalışmanın 2026 tarihli bir takip analizi [21], daha hafif bir "
    "hibrit-geçiş testi (DDCG) önererek aynı soruyu güncel deneylerle yeniden sınar. ABPT [22], ödülün "
    "yalnızca kısmen diferansiyellenebilir olduğu durumlar için düzeltilmiş bir BPTT şeması önerir."
)
H2("2.5 Temel Pekiştirmeli Öğrenme Algoritmaları")
P(
    "Karşılaştırmanın iki gradyansız çıpası: Schulman ve ark. [6]'nın PPO'su ve Haarnoja ve ark. [8,9]'ın "
    "Soft Actor-Critic'i. PPO'nun avantaj kestirimi, TRPO [17] ve Genelleştirilmiş Avantaj Kestirimi (GAE) "
    "[18]'e dayanır; SAC'ın maksimum-entropi çerçevesinin kökeni olan Soft Q-learning [19], entropi "
    "düzenlileştirmesinin off-policy öğrenmede neden GSAC'ın kaotik rejimde daha dayanıklı kaldığını "
    "açıklayan teorik zemini sağlar."
)
H2("2.6 Akışkanlar Dinamiğinde RL Tabanlı Aktif Akış Kontrolü (Bağlam)")
P(
    "Rabault ve ark. [23]'ın kurucu çalışmasından beri, gradyansız RL'nin türbülanslı akış kontrolünde "
    "başarılı olduğu tekrar tekrar gösterilmiştir: türbülanslı ayrılma kabarcığı kontrolü [10] klasik "
    "periyodik zorlamaya göre %9'a karşı %6.8 azaltım sağlar; erken bir çalışma [11] DRL'nin türbülanslı "
    "koşullara uygulanabilirliğini gösterir; gerçek zamanlı bluff-body iz kontrolü [24] rüzgar tüneli "
    "deneylerinde model-tabanlı temel çizgilere göre 2–4× performans elde eder. Bu kaynaklar hiçbiri "
    "analitik gradyan enjeksiyonunu kaotik/türbülanslı rejimle sistematik olarak karşılaştırmaz — bu "
    "çalışmanın hedeflediği boşluk tam olarak burasıdır."
)

# ============================================================ 3. YÖNTEM =====
H1(3, "Yöntem")
H2("3.1 Ortam")
P(
    "Deneyler, gerçek `hydrogym.jax.envs.kolmogorov.KolmogorovFlow` ortamında yürütülmüştür: 64×64 ızgara "
    "(4096 serbestlik derecesi), pseudo-spektral IMEX-RK4CN zaman entegrasyonu [1,13–16], 4 aktüatör modu. "
    "Bu çalışma, kütüphanenin resmi GPPO uygulamasını değil, aynı mekanizmayı (§3.3) kendi PPO/SAC eğitim "
    "döngümüz içinde yeniden uygulayan bağımsız bir kod tabanı kullanır; makalenin GPPO için kullandığı tam "
    "karışım oranı, ufuk uzunluğu ve hiperparametreler yayımlanmış metinde/şekil başlıklarında "
    "verilmediğinden, §3.3–3.4'te tanımlanan seçimler **kendi yeniden-uygulamamıza** aittir."
)
P(
    "Kütüphanenin `step_env` fonksiyonu, çağrı sırasında geçirilen `params.dt`/`params.action_time` "
    "değerlerini yok sayıp bunun yerine `self.default_params`'ı tekrar okur; bu yüzden zaman adımı ve eylem "
    "süresini POC ölçeğine indirmek için `default_params` özelliğini override eden bir alt sınıf "
    "(`FastKolmogorovFlow`) yazılmıştır. Kütüphanenin varsayılan `reward_alpha=0` değeri ödülü yalnızca "
    "kontrol maliyetine indirger; bu çalışmada `reward_alpha=1` kullanılarak akış enerjisi de ödüle dahil "
    "edilmiştir."
)
P(
    "<b>Reynolds sayısı seçimi.</b> Bu grid/dt kombinasyonunda (64×64, dt=0.001), kütüphanenin kendi "
    "çözücüsünün — sıfır eylemle bile — ampirik olarak Re≈350–380 civarında sayısal olarak ıraksadığı "
    "gözlemlenmiştir. Bu nedenle \"kaotik\" rejim için orijinal hedef olan Re=450 yerine, bu sınırın "
    "altında kalan ve Re=45'le anlamlı bir kontrast oluşturan en yüksek değer olan **Re=300** kullanılmıştır."
)
H2("3.2 PPO ve GAE")
P(
    "Politika, ortalama eylem μ<sub>θ</sub>(s) üreten bir MLP (2 gizli katman, 64 birim, tanh) ve "
    "durumdan bağımsız, öğrenilebilir bir log σ ile parametrelenmiş diyagonal Gauss dağılımıdır. Avantaj "
    "kestirimi GAE [18] ile yapılır (γ=0.99, λ=0.95):"
)
P(
    '<font face="DejaVuSansMono" size="8.6">Â<sub>t</sub> = Σ<sub>l=0..T-t-1</sub> (γλ)<sup>l</sup> δ<sub>t+l</sub>'
    ',   δ<sub>t</sub> = r<sub>t</sub> + γV(s<sub>t+1</sub>) − V(s<sub>t</sub>)</font>', "Body9"
)
P(
    "PPO'nun TRPO'nun [17] güven-bölgesi fikrinden türeyen kırpılmış vekil amaç fonksiyonu [6] (ε=0.2):"
)
P(
    '<font face="DejaVuSansMono" size="8.6">L<sup>PPO</sup>(θ) = E<sub>t</sub>[min(ρ<sub>t</sub>(θ)Â<sub>t</sub>, '
    'clip(ρ<sub>t</sub>(θ), 1−ε, 1+ε)Â<sub>t</sub>)],   ρ<sub>t</sub>(θ) = π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) '
    '/ π<sub>θold</sub>(a<sub>t</sub>|s<sub>t</sub>)</font>', "Body9"
)
P("Toplam kayıp, değer kaybı ve entropi bonusuyla düzenlileştirilir: L = −L<sup>PPO</sup> + 0.5 L<sup>V</sup> − 0.001 H.")

H2("3.3 GPPO: Analitik Gradyan Enjeksiyonu")
P(
    "SHAC [3] ve AHAC [5]'in kısa-ufuk fikrini izleyerek, H=6 adımlık kısa bir ufukta, politikanın "
    "**ortalama** eylemi (örnekleme yok) kullanılarak diferansiyellenebilir simülatör üzerinden ileri "
    "yayılım yapılır ve toplam ödülün negatifi θ'ya göre `jax.grad` ile analitik olarak türevlenir:"
)
P(
    '<font face="DejaVuSansMono" size="8.6">L<sub>analitik</sub>(θ) = −Σ<sub>t=1..H</sub> r(s<sub>t</sub>, '
    'μ<sub>θ</sub>(s<sub>t</sub>)),   g<sub>analitik</sub> = ∇<sub>θ</sub> L<sub>analitik</sub>(θ)</font>', "Body9"
)
P(
    "Nihai gradyan, standart PPO gradyanı g<sub>PPO</sub> = ∇<sub>θ</sub>L(θ) ile eşit ağırlıkla "
    "karıştırılır: <b>g<sub>final</sub> = 0.5 g<sub>analitik</sub> + 0.5 g<sub>PPO</sub></b> — Suh ve ark. "
    "[20]'ın önerdiği α-dereceli karma kestiricinin (α=0.5) özel bir hâli. Bu, HydroGym makalesinin GPPO'yu "
    "tanımlayışıyla [1] aynı mekanizmadır; ufuk uzunluğu ve karışım ağırlığı bu çalışmanın kendi "
    "seçimidir (bkz. §3.1, §7)."
)
H2("3.4 SAC ve GSAC")
P(
    "SAC [8,9], sıkıştırılmış (tanh) Gauss aktör, ikiz Q-kritik ve hedef ağlarla, Soft Q-learning [19]'in "
    "kökenlendirdiği maksimum-entropi hedefini optimize eder:"
)
P(
    '<font face="DejaVuSansMono" size="8.6">J(π) = E[Σ<sub>t</sub> r(s<sub>t</sub>,a<sub>t</sub>) + '
    'α H(π(·|s<sub>t</sub>))]</font>', "Body9"
)
P(
    "Kritikler, yumuşak hedef ağlarla (τ=0.005) güncellenir. <b>GSAC</b>, SAPO [4]'nun temel motivasyonunu "
    "(off-policy + maksimum-entropi + analitik gradyan) izleyerek, GPPO ile aynı mekanizmayı (§3.3, H=6, "
    "deterministik eylem) aktörün gradyanına uygular: <b>g<sub>final</sub> = 0.5 g<sub>analitik</sub> + 0.5 "
    "g<sub>π</sub></b>. GSAC, bu çalışmada SAPO ailesinin akışkanlar dinamiğine ilk taşınmasıdır; SAPO'nun "
    "kendisinden farklı olarak Rewarped platformu yerine HydroGym'in Kolmogorov akışı ortamını, ve SAPO'nun "
    "kendi entropi-tabanlı pürüzsüzleştirme şeması yerine sabit %50-%50 karışımı kullanır (bkz. §7)."
)

story.append(PageBreak())
# ============================================================ 4. DENEYSEL KURULUM ==
H1(4, "Deneysel Kurulum")
mktable(
    ["Parameter", "Value"],
    [
        ["Grid", "64×64 (4096 DOF)"],
        ["Solver", "pseudo-spectral IMEX-RK4CN (real `hydrogym.jax`)"],
        ["Re (regular / chaotic)", "45 / 300"],
        ["Episodes / steps per episode", "500 / 15"],
        ["Analytic horizon (GPPO/GSAC)", "H=6 steps"],
        ["Gradient mixing ratio", "50% standard + 50% analytic"],
        ["Learning rate", "3×10⁻⁴ (Adam)"],
        ["Network architecture", "2×64-unit MLP, tanh"],
        ["Seeds", "n=2 (seed=0, seed=1), see §5.2, §5.3, §7"],
        ["Hardware", "RTX 5060 laptop GPU, JAX + CUDA 12"],
    ],
    [5.5, 11.5],
    caption="Tablo 1. Deneysel kurulum parametreleri.",
)
P("Tüm dört algoritma, her iki Re rejiminde, HydroGym makalesinin bildirdiği eğitim bütçesiyle eşleşecek şekilde 500 episode ile eğitilmiştir.")

# ============================================================ 5. BULGULAR ===
H1(5, "Bulgular")
H2("5.1 Re=45 (düzenli akış): örnekleme verimliliği")
P(
    "GPPO, PPO'nun 500 episode boyunca ulaştığı en iyi düzeltilmiş (smoothed) ödül seviyesine seed=0'da "
    "**episode 162**'de ulaşırken, PPO aynı seviyeye ancak **episode 438**'de ulaşmıştır — **%63 episode "
    "tasarrufu**, makalenin bildirdiği **%65**'e [1] çarpıcı derecede yakın. Seed=1'de aynı karşılaştırma "
    "**%77 episode tasarrufu** vermiştir (GPPO ep50, PPO ep221) — mutlak sayılar farklı olsa da, GPPO'nun "
    "PPO'ya göre örnekleme avantajının **yönü** her iki seed'de de tutarlıdır."
)
mktable(
    ["Algorithm", "Episode reaching PPO's best reward (seed=0 / seed=1)", "Episode savings vs. PPO (seed=0 / seed=1)", "Final reward, last-5-ep. mean (seed=0 / seed=1)"],
    [
        ["PPO", "438 / 221", "— (reference)", "−25.1 / −25.2"],
        ["GPPO", "162 / 50", "63% / 77%", "−25.4 / −24.6"],
        ["SAC", "reached earlier (off-policy)", "—", "−16.7 / −16.9"],
        ["GSAC", "reached earlier (off-policy)", "—", "−18.0 / −17.1"],
    ],
    [2.6, 5.8, 4.2, 4.4],
    caption="Tablo 2. Re=45'te örnekleme verimliliği ve nihai performans karşılaştırması (seed=0 ve seed=1).",
)
fig("plots/all4_500ep_comparison.png", "Şekil 1. 4 algoritma, 500 episode ödül eğrileri — sol: Re=45 (seed=0), sağ: Re=300 ilk-NaN işaretli (seed=0).")
P(
    "SAC/GSAC'ta ise (off-policy replay-buffer yapısı zaten çok daha verimli olduğundan) GPPO-tarzı bir "
    "örnekleme-hızlandırma etkisi gözlenmemiştir: **GSAC, SAC'ın nihai performansını bu ayarlarda hiçbir "
    "seed'de geçememiştir.**"
)

H2("5.2 Re=300 (kaotik akış): kırılma noktası dört algoritmada da var, ama seed'e göre çok farklı bir desende")
P(
    "Kaotik rejimde **dört algoritmanın da bir kırılma noktası vardır** — hepsi 500 episode'u kararlı "
    "tamamlayamamış, er ya da geç sayısal olarak ıraksamıştır (NaN). Bu çalışmanın en önemli bulgusu, bu "
    "kırılma zamanının **seed'e göre de** çarpıcı biçimde değiştiğidir:"
)
mktable(
    ["Algorithm", "Breaking point, seed=0", "Delay vs. PPO, seed=0", "Breaking point, seed=1", "Delay vs. PPO, seed=1"],
    [
        ["PPO", "12", "1.0× (ref.)", "35", "1.0× (ref.)"],
        ["SAC", "36", "3.0×", "6", "0.17× (earlier than PPO)"],
        ["GPPO", "83", "6.9×", "40", "1.14×"],
        ["GSAC", "141", "11.8×", "63", "1.8×"],
    ],
    [3.0, 3.6, 4.0, 3.6, 3.8],
    caption="Tablo 3. Re=300'de kırılma noktası, seed=0 ve seed=1 karşılaştırması. Bkz. Şekil 5 (iki-seed karşılaştırması), Şekil 2 (seed=0) ve Şekil 1 sağ panel (seed=0).",
)
fig("plots/seed_comparison.png", "Şekil 5. Sol: Re=300'de ilk-NaN episode'u, seed=0 vs. seed=1 (gruplu çubuklar). Sağ: aynı veri PPO'ya göre gecikme çarpanı olarak — SAC'ın yön değişimini ve GSAC'ın her iki seed'de de en dayanıklı kalışını görsel olarak gösterir.")
fig("plots/stability_delay_500ep.png", "Şekil 2. Re=300'de ilk NaN episode'u — algoritmaya göre kararsızlığa düşme süresi, PPO'ya göre kaç kat (seed=0).", width_cm=10, ratio=6.07/8.5)
P(
    "Seed=0'da bu tablo neredeyse çarpımsal bir yapı sergilemekteydi: off-policy yapı tek başına (SAC/PPO) "
    "3.0× gecikme sağlarken, analitik gradyan tek başına on-policy zeminde (GPPO/PPO) 6.9×, off-policy "
    "zeminde (GSAC/SAC) 3.9× gecikme sağlıyor, ikisi birlikte (GSAC) 11.8×'e ulaşıyordu (3.0×3.9≈11.7). "
    "**Bu desen seed=1'de tekrarlanmamıştır.** Seed=1'de SAC, PPO'dan **daha erken** (episode 6'da, "
    "PPO'nun episode 35'ine karşı) çökmüştür — off-policy yapının tek başına kararlılığı artırdığı iddiası "
    "burada **tersine dönmektedir**. Buna karşın iki yön tutarlı kalmaktadır:"
)
bullet("**Analitik gradyan enjeksiyonu her iki seed'de de geciktiricidir:** GPPO, PPO'dan daha geç kırılır (6.9× ve 1.14×); GSAC, SAC'tan daha geç kırılır (3.9× ve 10.5×) — büyüklük çok değişse de yön değişmemektedir.")
bullet("**GSAC, dört yöntem içinde her iki seed'de de en dayanıklısıdır** (seed=0: 141 episode; seed=1: 63 episode — ikisi de kendi seed'indeki en yüksek değer). Bunun ayrıntılı mekanik açıklaması §5.3'te sunulmaktadır.")
P(
    "Kısacası: n=1'in önerdiği \"temiz, çarpımsal, iki bağımsız mekanizma\" hikâyesi fazla iyimserdi. n=2 "
    "ile ortaya çıkan daha gerçekçi bulgu, off-policy yapının kararlılığa katkısının **kendi başına "
    "güvenilir olmadığı**, yalnızca analitik gradyanla **birleştiğinde** tutarlı bir fayda sağladığıdır."
)

H2("5.3 Kırılmanın mekanizması: ani tetiklenme ve tampon-seyreltme etkisi")
P(
    "Yukarıdaki tablo *ne zaman* kırıldığını gösteriyor; bu alt bölüm, ham eğitim eğrilerini doğrudan "
    "inceleyerek *neden* kırıldığına dair somut bulgular sunar."
)
P("<b>(a) Kırılma önce sinyal vermeden, ani gerçekleşiyor.</b> Her dört algoritma için de, her iki seed'de de, kırılma öncesi 40 episode'luk pencerede ödül ortalaması ve standart sapması kararlıdır:")
mktable(
    ["Algorithm", "First-half mean (seed=0 / seed=1)", "Second-half mean (seed=0 / seed=1)", "Std. dev. (seed=0 / seed=1)"],
    [
        ["PPO", "−32.75 / −32.28", "−32.77 / −33.05", "1.35 / 1.32"],
        ["SAC", "−33.31 / −32.76", "−32.50 / −32.79", "1.30 / 1.39"],
        ["GPPO", "−32.61 / −32.38", "−32.98 / −33.01", "1.10 / 1.24"],
        ["GSAC", "−32.49 / −32.19", "−32.46 / −32.08", "1.32 / 1.16"],
    ],
    [2.6, 5.4, 5.4, 3.6],
    caption="Tablo 4. Kırılma öncesi 40-episode'luk penceredeki ödül istatistikleri (seed=0 ve seed=1). Hiçbir algoritmada/seed'de ısınma veya kademeli bozulma yok.",
)
P(
    "Bu, kırılmanın politikanın *kademeli olarak* kontrol kuvvetini artırıp çözücüyü zorlamasından değil — "
    "**tek bir episode'un rastgele örneklenen aksiyon dizisinin**, o anki politika ile birleşince akışı "
    "çözücünün Re≈350–380 civarındaki sayısal kararlılık sınırının ötesine iten, ani ve stokastik bir "
    "olaydan kaynaklandığını düşündürür — Metz ve ark. [7]'ın kaos-kaynaklı gradyan patlamasının ve "
    "Boffetta-Musacchio [25]'nun Re ile daralan öngörü ufkunun beklediği türden bir \"kelebek etkisi\" "
    "tetiklenmesiyle tutarlıdır."
)
P("<b>(b) On-policy ve off-policy algoritmalar tetiklenme sonrası farklı davranır — tampon-seyreltme etkisi.</b> Tetiklenme anıyla kalıcı çöküş anı arasındaki mesafe:")
mktable(
    ["Algorithm", "Trigger", "Permanent collapse", "Recovered ep."],
    [["PPO", "ep 12", "ep 12", "0"], ["SAC", "ep 36", "ep 36", "0"], ["GPPO", "ep 83", "ep 83", "0"], ["GSAC", "ep 141", "ep 144", "2 (ep 142,143)"]],
    [3.0, 3.4, 3.4, 4.0],
    caption="Tablo 5a. Tetiklenme vs. kalıcı çöküş episode'u — seed=0.",
)
mktable(
    ["Algorithm", "Trigger", "Permanent collapse", "Recovered ep."],
    [["PPO", "ep 35", "ep 35", "0"], ["SAC", "ep 6", "ep 6", "0"], ["GPPO", "ep 40", "ep 40", "0"], ["GSAC", "ep 63", "ep 63", "0"]],
    [3.0, 3.4, 3.4, 4.0],
    caption="Tablo 5b. Tetiklenme vs. kalıcı çöküş episode'u — seed=1. Hiçbir algoritmada kurtarma penceresi yok (§5.3(c) bunun neden beklendiğini açıklıyor).",
)
fig("plots/mechanism_analysis.png", "Şekil 3. Sol: tetiklenme vs. kalıcı-çöküş episode'u (seed=0). Sağ: tetiklenme anındaki tamponun zehirli-geçiş oranı (off-policy algoritmalar, seed=0).")
P(
    "PPO/GPPO'da (on-policy), tetiklenme ve kalıcı çöküş **aynı anda** olur — kaçış yolu yoktur. SAC'ta da "
    "tetiklenme aynı episode'da kalıcılaşır, çünkü episode 36'da tamponda yalnızca ~540 geçiş vardır ve bu "
    "episode'un ~15 \"zehirli\" geçişi tamponun **%2.8**'ini oluşturur; 640 çekilişte bu geçişlerden "
    "hiçbirinin örneklenmeme olasılığı pratikte sıfırdır. GSAC'ta tetiklenme çok daha geç (episode 141) "
    "gerçekleştiği için tampon çok daha büyüktür (~2115 geçiş) ve zehirli geçiş oranı yalnızca **%0.7** — bu, "
    "hayatta kalma olasılığını ~%1'e çıkarır ve gözlemlenen 2-episode'luk \"hayatta kalma\" ile "
    "niteliksel olarak tutarlıdır."
)
mktable(
    ["Algorithm", "Trigger", "Buffer", "Bad-fraction", "Survival prob."],
    [["SAC", "ep 36", "~540 transitions", "2.8%", "≈0%"], ["GSAC", "ep 141", "~2115 transitions", "0.7%", "≈1%"]],
    [2.6, 2.8, 3.9, 3.1, 3.6],
    caption="Tablo 6a. Tetiklenme anındaki tampon-seyreltme hesabı — seed=0.",
)
mktable(
    ["Algorithm", "Trigger", "Buffer", "Bad-fraction", "Survival prob."],
    [["SAC", "ep 6", "~90 transitions", "16.7%", "≈0%"], ["GSAC", "ep 63", "~945 transitions", "1.6%", "≈0.004%"]],
    [2.6, 2.8, 3.9, 3.1, 3.6],
    caption="Tablo 6b. Tetiklenme anındaki tampon-seyreltme hesabı — seed=1, mekanizma çapraz-doğrulaması.",
)
P(
    "<b>(c) Seed=1 ile çapraz-doğrulama.</b> Bu mekanik model, seed=0 verisiyle kurulduktan sonra **seed=1 "
    "üzerinde bağımsız olarak test edilmiştir** ve doğru öngörüde bulunmuştur: seed=1'de hem SAC hem GSAC "
    "çok daha erken tetiklendiği için tamponları çok daha küçüktür, hesaplanan hayatta-kalma olasılığı her "
    "ikisinde de ~%0'a yakındır — nitekim seed=1'de **hiçbir algoritmada** bir kurtarma penceresi "
    "gözlenmemiştir. Mekanizmanın kendisi, seed=0'da GSAC'ın kurtarmasını **ve** seed=1'de hiçbir "
    "algoritmanın kurtarılamamasını **her ikisini de** doğru öngörmektedir — bu, §5.2'deki ham gecikme "
    "çarpanlarının aksine, **n=2 ile nicel olarak doğrulanmış** bir bulgudur."
)
P(
    "<b>Açık soru:</b> Bu analiz tetiklenme-sonrası kilitlenme mekanizmasını açıklıyor ve iki seed arasında "
    "doğrulanmıştır, ama *tetiklenme episode'unun kendisinin* neden algoritmaya ve seed'e göre bu denli "
    "değiştiğini açıklamıyor — bunun için eylem genliğinin episode-episode nasıl evrildiğinin ayrıca "
    "loglanması gerekir; bu, doğal bir sonraki tanı deneyi olarak §7'de not edilmiştir."
)

story.append(PageBreak())
# ============================================================ 6. TARTIŞMA ===
H1(6, "Tartışma ve Katkı")
P("Bu çalışma, HydroGym makalesinin Kolmogorov akışı üzerindeki GPPO-vs-PPO sonucunu [1] iki yönde genişletir:")
bullet(
    "**Rejim genişlemesi:** Re=45'te makale her iki seed'de de yön olarak doğrulanır (%63 ve %77 vs "
    "makalenin bildirdiği %65); Re=300'de makalenin göstermediği bir bulgu ortaya çıkar — GPPO'nun "
    "avantajı \"daha az episode'da daha iyi ödül\"den \"daha uzun süre kararlı kalabilme\"ye dönüşür, ve bu "
    "avantaj da sonunda tükenir."
)
bullet(
    "**Mekanizma ayrıştırması (n=2 ile düzeltilmiş):** n=1 ile ilk izlenim, off-policy ve analitik gradyan "
    "katkılarının bağımsız ve yaklaşık çarpımsal olduğu yönündeydi; **ikinci bir seed bu izlenimi "
    "düzeltmiştir.** Analitik gradyanın geciktirici etkisi her iki seed'de de yön olarak tutarlı kalırken, "
    "off-policy yapının **tek başına** geciktirici olduğu iddiası seed=1'de tersine dönmüştür. Yalnızca "
    "**GSAC'ın her iki seed'de de en dayanıklısı olması** tutarlı kalmıştır; bu, off-policy yapının katkısının "
    "kendi başına değil, analitik gradyanla birleştiğinde güvenilir olduğunu düşündürmektedir."
)
P(
    "Bulgular, Metz ve ark. [7]'ın genel kaos-kaynaklı gradyan kararsızlığı gözlemini ve "
    "Boffetta-Musacchio [25]'nun Re-bağımlı öngörülemezlik sonucunu somut bir akışkanlar dinamiği kontrol "
    "görevine taşır. n=2 ile ortaya çıkan tablo, kaotik rejimdeki kararlılığın **hem algoritma tasarımına "
    "hem de eğitimin rastgele seyrine (seed'e)** bağlı olduğunu göstermektedir. Buna karşın §5.3'teki "
    "tampon-seyreltme mekanizması, tetiklenme-sonrası davranışı her iki seed'de de doğru öngörerek, bu "
    "çalışmanın n=2 ile nicel olarak sınanmış ve doğrulanmış tek mekanizmik bulgusunu oluşturur. Bulgular, "
    "HydroGym makalesinin ana iddiasını (GPPO'nun faydalı olduğu) çürütmez; tersine bunu daha geniş bir "
    "algoritma ailesi, rejim uzayı ve **birden fazla seed** üzerinde sınar."
)

# ============================================================ 7. SINIRLAMALAR ==
H1(7, "Sınırlamalar")
bullet(
    "**n=2 seed (seed=0, seed=1)** — istatistiksel anlamda hâlâ küçük bir örneklem; n=2'nin asıl değeri, "
    "n=1 ile görülen bir deseni (çarpımsallık) çürütebilmesidir — nitekim burada da öyle olmuştur (§5.2). "
    "Daha fazla seed (3–5), SAC'ın seed=0/seed=1 arasındaki yön değişikliğinin hangi ucun \"tipik\" "
    "olduğunu ayırt etmek için gereklidir; bu çalışmanın kapsamı şimdilik n=2 ile sınırlandırılmıştır."
)
bullet(
    "**Re=300'deki gecikme çarpanları seed'e göre büyük varyans gösteriyor** (GPPO/PPO: 1.14×–6.9×; "
    "GSAC/SAC: 3.9×–10.5×; SAC/PPO: 0.17×–3.0×, yön değişimiyle) — bunlar \"yasa\" değil, iki "
    "nokta-gözlemdir. Tek istikrarlı bulgu GSAC'ın her iki seed'de de en dayanıklı algoritma olmasıdır."
)
bullet("**Re=300, orijinal hedef Re=450 yerine kullanıldı** — kütüphanenin çözücüsünün bu grid/dt kombinasyonunda Re≈350–380 civarında kararlılık sınırına sahip olduğu keşfedilmiştir.")
bullet("**GPPO/GSAC'ın karışım oranı, ufuk uzunluğu ve diğer hiperparametreleri kendi yeniden-uygulamamıza aittir** — HydroGym makalesi bu değerleri paylaşmamaktadır; %63/%77'lik sonuçların %65 ile aynı büyüklük mertebesinde olması güçlü bir işaret olsa da, birebir aynı olduğu iddia edilmemektedir.")
bullet("**Kısa episode'lar (15 RL-adımı) ve kısa analitik ufuk (6 adım)** — hem RL yakınsaması hem kaotik ayrışmanın tam gözlemlenmesi için yetersiz olabilir.")
bullet("**GSAC'ın karışım ağırlığı sabit %50-%50** — Suh ve ark. [20]'ın adaptif α-dereceli kestiricisi, AHAC [5]'in adaptif-ufuk fikri veya SAPO [4]'nun entropi-tabanlı pürüzsüzleştirmesi denenmemiştir.")
bullet("**Tetiklenme-episode'unun kendisinin nedeni açık kalıyor** — bu, n=2 ile de çözülmeyen tek açık soru olarak kaldı; eylem genliği loglu bir takip deneyi gerekmektedir.")
bullet(
    "**§5.3'teki tampon-seyreltme mekanizması artık n=2 ile sınanmıştır ve doğrulanmıştır** — seed=0'da "
    "kurulan model, seed=1'in tamamen farklı tetiklenme profilini doğru biçimde öngörmüştür. Bu, çalışmanın "
    "n=2 ile gerçekten doğrulanmış tek mekanizmik iddiasıdır; yine de n=2 bu modelin genel geçerliliğini "
    "kesin olarak kanıtlamaz."
)

# ============================================================ 8. SONUÇ ======
H1(8, "Sonuç ve Gelecek Çalışma")
P(
    "Analitik gradyan enjeksiyonu, kaotik rejimde sayısal kararsızlığı ortadan kaldırmaz — **erteler**. İki "
    "bağımsız eğitim seed'i üzerinden test edildiğinde, bu ertelemenin **yönü** tutarlı kalırken, "
    "**büyüklüğü** seed'e göre büyük ölçüde değişmektedir; off-policy yapının tek başına koruyucu olduğu "
    "iddiası ikinci seed'te tersine dönmüştür. Buna karşın, **GSAC'ın dört yöntem içinde her iki seed'de de "
    "en dayanıklı konfigürasyon olması** istikrarlı kalmıştır. Ayrıca, kırılma sonrası davranışı açıklayan "
    "tampon-seyreltme mekanizması, seed=0 üzerinde kurulup seed=1 üzerinde bağımsız olarak doğrulanmıştır."
)
P(
    "<b>Bu tur için kapsam kararı:</b> Kullanılan hesaplama bütçesi göz önüne alındığında, bu taslak için "
    "n=2 (seed=0, seed=1) yeterli kabul edilmiş ve üçüncü bir seed'e (seed=2) geçilmemiştir. Gelecekteki "
    "planlanan işler:"
)
bullet("**3–5 seed'e genişletmek** ve biçimsel bir istatistiksel anlamlılık testi (permütasyon/bootstrap) uygulamak.")
bullet("**Tetiklenme-episode'unun kendisinin nedenini** açıklığa kavuşturmak için eylem-genliği loglu bir takip deneyi.")
bullet("GSAC'ın karışım ağırlığını AHAC [5]/SAPO [4] tarzı adaptif bir şemaya taşımak; Boffetta-Musacchio [25] çerçevesini kullanarak gecikme çarpanlarının Re'ye karşı nicel bir ölçeklenme yasasına oturup oturmadığını test etmek.")

fig("plots/paper_comparison_500ep.png", "Şekil 4. PPO vs. GPPO, HydroGym makalesinin Şekil 5'iyle doğrudan karşılaştırma formatı — Re=45 (makalenin rejimini yeniden üretir) ve Re=300 (bu çalışmanın uzantısı, orijinal makalede test edilmemiştir).")

story.append(PageBreak())
# ============================================================ KAYNAKLAR =====
H1(None, "Kaynaklar")
refs = [
"[1] C. Lagemann, S. Mokbel, M. Gondrum, M. Rüttgers, Y. Wang, P. Suárez, L. Paehler, D. A. Bezgin, A. B. Buhendwa, J. L. Callaham, S. Ahnert, N. Zolman, X. Shao, J.-C. Loiseau, N. Adams, M. Meinke, W. Schröder, K. Lagemann, E. Lagemann, R. Vinuesa, S. L. Brunton, \"The HydroGym Reinforcement Learning Platform for Fluid Dynamics,\" <i>Nature</i>, 2026. doi:10.1038/s41586-026-10917-6. Ön baskı: arXiv:2512.17534.",
"[2] J. Becktepe, A. Franz, N. Thuerey, S. Peitz, \"Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control (FluidGym),\" arXiv:2601.15015, 2026.",
"[3] J. Xu, V. Makoviychuk, Y. Narang, F. Ramos, W. Matusik, A. Garg, M. Macklin, \"Accelerated Policy Learning with Parallel Differentiable Simulation (SHAC),\" in Proc. ICLR, 2022. arXiv:2204.07137.",
"[4] E. Xing, V. Luk, J. Oh, \"Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation (SAPO / Rewarped),\" in Proc. ICLR (Spotlight), 2025. arXiv:2412.12089.",
"[5] I. Georgiev, K. Srinivasan, J. Xu, E. Heiden, A. Garg, \"Adaptive Horizon Actor-Critic for Policy Learning in Contact-Rich Differentiable Simulation (AHAC),\" in Proc. ICML, 2024. arXiv:2405.17784.",
"[6] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, \"Proximal Policy Optimization Algorithms,\" arXiv:1707.06347, 2017.",
"[7] L. Metz, C. D. Freeman, S. S. Schoenholz, T. Kachman, \"Gradients are Not All You Need,\" arXiv:2111.05803, 2021.",
"[8] T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, \"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,\" in Proc. ICML, 2018. arXiv:1801.01290.",
"[9] T. Haarnoja, A. Zhou, K. Hartikainen ve ark., \"Soft Actor-Critic Algorithms and Applications,\" arXiv:1812.05905, 2018.",
"[10] B. Font, F. Alcántara-Ávila, J. Rabault, R. Vinuesa, O. Lehmkuhl, \"Deep Reinforcement Learning for Active Flow Control in a Turbulent Separation Bubble,\" <i>Nature Communications</i>, 2025. doi:10.1038/s41467-025-56408-6.",
"[11] \"Applying Deep Reinforcement Learning to Active Flow Control in Turbulent Conditions,\" arXiv:2006.10683, 2020.",
"[12] \"Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes,\" Annual Review of Control, Robotics, and Autonomous Systems, 2025.",
"[13] J. Bradbury, R. Frostig, P. Hawkins ve ark., \"JAX: Composable Transformations of Python+NumPy Programs,\" 2018. github.com/jax-ml/jax.",
"[14] D. Kochkov, J. A. Smith, A. Alieva, Q. Wang, M. P. Brenner, S. Hoyer, \"Machine Learning–Accelerated Computational Fluid Dynamics (JAX-CFD),\" PNAS 118(21), 2021. doi:10.1073/pnas.2101784118.",
"[15] D. A. Bezgin, A. B. Buhendwa, N. A. Adams, \"JAX-Fluids: A Fully-Differentiable High-Order CFD Solver for Compressible Two-Phase Flows,\" arXiv:2203.13760, 2022.",
"[16] D. A. Bezgin, A. B. Buhendwa, N. A. Adams, \"JAX-Fluids 2.0: Towards HPC for Differentiable CFD of Compressible Two-Phase Flows,\" arXiv:2402.05193, 2024.",
"[17] J. Schulman, S. Levine, P. Moritz, M. I. Jordan, P. Abbeel, \"Trust Region Policy Optimization (TRPO),\" arXiv:1502.05477, 2015.",
"[18] J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, \"High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE),\" arXiv:1506.02438, 2015 (ICLR 2016).",
"[19] T. Haarnoja, H. Tang, P. Abbeel, S. Levine, \"Reinforcement Learning with Deep Energy-Based Policies (Soft Q-Learning),\" in Proc. ICML, 2017. arXiv:1702.08165.",
"[20] H. J. T. Suh, M. Simchowitz, K. Zhang, R. Tedrake, \"Do Differentiable Simulators Give Better Policy Gradients?,\" in Proc. ICML, 2022. arXiv:2202.00817.",
"[21] K. Onoda, P. Parmas, M. Yaguchi, Y. Matsuo, \"Does 'Do Differentiable Simulators Give Better Policy Gradients?' Give Better Policy Gradients?,\" arXiv:2604.18161, 2026.",
"[22] F. Li, F. Sun, T. Zhang, D. Zou, \"ABPT: Amended Backpropagation through Time with Partially Differentiable Rewards,\" arXiv:2501.14513, 2025.",
"[23] J. Rabault, M. Kuchta, A. Jensen, U. Réglade, N. Cerardi, \"Artificial Neural Networks Trained through Deep Reinforcement Learning Discover Control Strategies for Active Flow Control,\" Journal of Fluid Mechanics 865, 281–302, 2019. doi:10.1017/jfm.2019.62.",
"[24] \"Autonomous Real-Time Control of Turbulent Dynamics (REACT),\" Imperial College London, arXiv:2509.11002, 2026.",
"[25] G. Boffetta, S. Musacchio, \"Chaos and Predictability of Homogeneous-Isotropic Turbulence,\" Physical Review Letters 119, 054102, 2017. doi:10.1103/PhysRevLett.119.054102. arXiv:1707.01648.",
]
for r in refs:
    story.append(Paragraph(md(r), styles["RefBody"]))

SP(6)
P(
    "*Kaynakça 26 Ağustos 2026 tarihinde gerçek zamanlı web taraması ile doğrulanmıştır (başlık, yazar "
    "listesi ve yayın bilgisi her kaynağın kendi sayfasından teyit edilmiştir); ayrıca bkz. tamamlayıcı "
    "belge `HydroGym_Literatur_Taramasi_HaktanGuloglu.pdf` (N=25, 25 Ağustos 2026).*", "Small"
)

# ============================================================ ŞEKİLLER/TABLOLAR İNDEKSİ ==
H1(None, "Şekiller ve Tablolar İndeksi")
P(
    "Tüm şekil başlıkları, eksen etiketleri ve tablo içerikleri İngilizce olarak üretilmiştir (uluslararası "
    "okunabilirlik için); yalnızca çevreleyen düz metin Türkçedir."
)
bullet("**Şekil 1** — `all4_500ep_comparison.png`: 4 algoritma, Re=45 ve Re=300 ödül eğrileri (§5.1, §5.2).")
bullet("**Şekil 2** — `stability_delay_500ep.png`: Re=300'de ilk-NaN episode'u, PPO'ya göre gecikme çarpanı (§5.2, Tablo 3).")
bullet("**Şekil 3** — `mechanism_analysis.png`: tetiklenme vs. kalıcı-çöküş episode'u; tampon zehirli-geçiş oranı (§5.3, Tablo 5-6).")
bullet("**Şekil 4** — `paper_comparison_500ep.png`: PPO vs. GPPO, orijinal HydroGym makalesinin Şekil 5'iyle karşılaştırma (§6).")
bullet("**Şekil 5** — `seed_comparison.png`: seed=0 vs. seed=1 kırılma noktası ve gecikme çarpanı karşılaştırması (§5.2, Tablo 3).")
SP(4)
bullet("**Tablo 1** (§4) — Deneysel kurulum. **Tablo 2** (§5.1) — Re=45 örnekleme verimliliği. **Tablo 3** (§5.2) — Re=300 kırılma noktası. **Tablo 4** (§5.3) — Kırılma öncesi ödül istatistikleri. **Tablo 5a/5b** (§5.3) — Tetiklenme vs. kalıcı çöküş. **Tablo 6a/6b** (§5.3) — Tampon-seyreltme hesabı.")

doc.build(story)
print("wrote", OUT)
