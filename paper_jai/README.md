# HydroGym makalesi — *Journal of Automation and Intelligence* LaTeX paketi

Elsevier **CAS** şablonuna (`cas-sc.cls`, tek kolon) taşınmış hâli. İçerik
`../paper_tex/` ile aynı — `sections/` ve `refs.bib` bit-aynı kopyalandı,
sadece `main.tex` (preamble + ön-bilgi) CAS'a göre yeniden yazıldı.

## Derleme

```
latexmk -pdf main.tex          # veya
tectonic -X compile main.tex   # (bibtex'i otomatik çalıştırır)
```

Elle: `pdflatex main` → `bibtex main` → `pdflatex main` → `pdflatex main`.
Çıktı: `main.pdf` (16 sayfa).

### Overleaf
New Project → Upload Project → bu klasörün zip'i. Ana dosya `main.tex`,
derleyici **pdfLaTeX**. CAS sınıfı Overleaf'te hazır; zip'teki `.cls`/`.sty`/
`.bst` dosyaları da yerel kopya olarak yeter.

## Bu pakette ne var

| Dosya | Açıklama |
|---|---|
| `main.tex` | CAS preamble + `frontmatter` (başlık, yazar, `\affiliation`, abstract, `highlights`, `keywords`) + `\input{sections/..}` |
| `sections/01..08.tex` | Introduction → Conclusion — **sınıftan bağımsız**, `../paper_tex/` ile aynı |
| `refs.bib` | 30 kaynak (DOI'li) |
| `figures/*.pdf` | 6 sonuç şekli (`plot_paper_n5.py`); yöntem şekilleri `03_methods.tex` içinde TikZ |
| `cas-sc.cls`, `cas-dc.cls`, `cas-common.sty` | Elsevier CAS 2.4 sınıf dosyaları (yerel kopya) |
| `cas-model2-names.bst` | CAS yazar-yıl bib stili (kullanılmıyor; yedek) |
| `elsarticle-num.bst` | **kullanılan** numaralı (Vancouver) bib stili |
| `thumbnails/` | CAS ön-bilgi ikonları (e-posta vb.) — `\ead` bunları ister |

## CAS'a özgü kararlar

- **Tek kolon** (`cas-sc`). Çift kolon dizgi için `main.tex`'teki
  `\documentclass` satırını `cas-dc` yap — `sections/` dosyaları değişmez.
- **Numaralı atıf**: `\usepackage[numbers,sort&compress]{natbib}` +
  `\bibliographystyle{elsarticle-num}`. Bölümlerdeki `\cite{...}` olduğu gibi
  çalışır (dönüştürme yapılmadı).
- `main.tex` içinde iki küçük CAS yaması var (`\ExplSyntaxOn` bloğu):
  (1) ilk-sayfa alt bilgisine "Elsevier" yerine dergi adını yazar;
  (2) ORCID verilmediğinde boş "ORCID(s):" dipnotunu bastırmaz.
  ORCID'ler `\author[n]{Ad}[orcid=...]` ile eklendi; sorumlu yazar Cem Özkurt (`\cormark`).
- `\credit{...}` + `\printcredits` → sonda **CRediT authorship contribution
  statement** bölümünü üretir.

## Bilinen zararsız uyarılar

- `algorithm2e.sty: Invalid UTF-8 byte` — tectonic paketindeki `.sty`
  kopyasından; dizgiyi etkilemez (Overleaf/TeX Live'da hiç çıkmaz).
- `Overfull \hbox (117pt) ... at \maketitle` — CAS ön-bilgi bloğundaki bir
  ayraç çizgisinin iç ölçümü; PDF'te görünür bir taşma yok (Elsevier'in kendi
  `cas-sc-sample` dosyası da aynı uyarıyı verir).

## Kaynak doğruluk

Düzyazı kaynağı hâlâ `../PAPER_DRAFT_EN.md`. İçerik güncellemesi orada +
`sections/` içinde yapılır; `../paper_tex/` (elsarticle) ve bu paket
(CAS/JAI) aynı `sections/`'tan beslenir.
