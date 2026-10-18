#!/bin/bash
# Regenerates the manuscript figures from results/*.json and converts the six
# result figures to plots/sonuc_1..6.png (needs pdftoppm from poppler-utils).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 analysis/plot_paper_n5.py
i=1
for f in re45_curves re45_final re300_curves re300_breakpoint mechanism gradnorm; do
  pdftoppm -png -r 150 -singlefile "paper/elsarticle/figures/fig_$f.pdf" "plots/sonuc_$i"
  i=$((i+1))
done
