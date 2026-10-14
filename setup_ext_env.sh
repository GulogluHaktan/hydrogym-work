#!/usr/bin/env bash
# Portable venv kurulumu — HydroGym ek çalışmalar paketi (A + B + C).
# venv'i BU KLASÖRÜN İÇİNE ("./venv") kurar, böylece zip'i nereye açarsan aç çalışır.
#
# Hedef makine gereksinimleri:
#   - Python 3.10 veya 3.11
#   - NVIDIA GPU + sürücü >= 550 (CUDA 12 wheel'leri sürücüyle gelir; sistem CUDA GEREKMEZ)
#   - ~8 GB boş disk (venv ~6 GB; 4.5 GB'ı CUDA wheel'leri)
#   - Ağ bağlantısı (ilk kurulumda pip + JAX bundle indirir)
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/venv"
echo "=== paket klasörü: $HERE"
echo "=== venv:          $VENV"

PY="${PYTHON:-python3}"
PYVER="$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "=== python:        $PY ($PYVER)"
case "$PYVER" in
  3.10|3.11) ;;
  *) echo "UYARI: orijinal koşular 3.11 ile yapıldı; $PYVER test edilmedi." ;;
esac

if [ ! -d "$VENV" ]; then
  echo "=== venv oluşturuluyor"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --no-cache-dir --upgrade pip wheel
echo "=== sabitlenmiş bağımlılıklar (requirements-lock.txt, 122 paket, ~6 GB)"
pip install --no-cache-dir -r "$HERE/requirements-lock.txt"
pip cache purge 2>/dev/null || true
# hydrogym paketi bu klasörde gömülü; run_ext_all.sh PYTHONPATH ile ekliyor,
# ayrıca pip install gerekmiyor (poetry-core bağımlılığı yok).
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

echo
echo "=== JAX GPU görüyor mu?"
python - <<'PY'
import jax
print("jax", jax.__version__, jax.devices())
if not any(d.platform == "gpu" for d in jax.devices()):
    raise SystemExit("!! GPU yok — CPU çok yavaş")
PY
echo "=== Kolmogorov env import testi"
python -c "from hydrogym.jax.envs.kolmogorov import KolmogorovFlow; print('KolmogorovFlow OK')"
echo
echo "=== kurulum tamam. Şimdi:  bash run_ext_all.sh"
