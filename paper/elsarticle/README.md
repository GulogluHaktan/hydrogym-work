# HydroGym makalesi — LaTeX kaynak paketi

**Derleme:** `main.tex` → `pdflatex` → `bibtex` → `pdflatex` × 2
(veya `latexmk -pdf main.tex`). `main.pdf` derlenmiş çıktıdır (24 sayfa).

## Overleaf ile (önerilen)
1. overleaf.com → **New Project → Upload Project** → bu klasörün zip'ini yükle.
2. Ana dosya: `main.tex`. Derleyici: **pdfLaTeX** (Menu → Compiler).
3. `elsarticle` sınıfı Overleaf'te hazır gelir; ek kurulum gerekmez.

## Yerelde
```
latexmk -pdf main.tex
```
Gerekli paketler: `elsarticle`, `amsmath`, `graphicx`, `booktabs`, `siunitx`,
`algorithm2e`, `tikz`, `subcaption`, `microtype`, `hyperref`, `cleveref`
(hepsi TeX Live / MiKTeX standart dağıtımında var).

## Dosya yapısı
| Dosya | İçerik |
|---|---|
| `main.tex` | preamble + frontmatter (başlık, özet, keywords) + `\input`'lar |
| `sections/01..08` | Introduction → Conclusion, sınıftan bağımsız |
| `refs.bib` | 30 kaynak, DOI'li (`elsarticle-num` stili) |
| `figures/*.pdf` | 6 sonuç şekli (vektör). Metod şekilleri TikZ ile `.tex` içinde. |

## Hedef dergi
Şu an `elsarticle` (Neural Networks / genel Elsevier). MLST (IOP) seçilirse
`\documentclass` → `iopart` olarak değişir; `sections/` dosyaları aynı kalır.
