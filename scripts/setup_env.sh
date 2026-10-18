#!/bin/bash
# Portable environment setup for the HydroGym analytic-gradient experiments.
# Reproduces the venv used for the seed=0/seed=1 sweeps on another machine
# (e.g. the atölye workstation, also an RTX 5060 -> same ~8.5 h / seed).
#
# Storage-constrained target? Clone shallow and WITHOUT the (unused, Firedrake)
# submodule, then run this script:
#
#   git clone --depth 1 <this-repo> HydroGymDeney
#   cd HydroGymDeney/hydrogym_real
#   bash scripts/setup_env.sh
#
# Requirements on the target machine:
#   - Python 3.10 or 3.11 (3.11 used for the original runs)
#   - An NVIDIA GPU + recent driver (>= 550, for CUDA 12.9 runtime wheels).
#     NO system CUDA toolkit needed — the nvidia-*-cu12 pip wheels bundle it.
#   - Disk: venv is ~6 GB (4.5 GB of that is bundled CUDA wheels — unavoidable
#     for GPU JAX; hydrogym transitively needs wandb/tfp/distrax too, so the
#     venv can't be trimmed much). This script uses --no-cache-dir so peak
#     usage during install stays ~6.5 GB instead of ~9-10 GB.
#     => need ~8 GB free, 10 GB comfortable.
#   - CPU-only fallback works but is ~10-30x slower (a seed sweep goes from
#     ~8.5 h to multiple days) — not recommended.
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
REPO_ROOT="$(cd "$HERE/.." && pwd)"
VENV="$REPO_ROOT/venv"

echo "=== repo root: $REPO_ROOT"
echo "=== venv:      $VENV"

PY="${PYTHON:-python3}"
PYVER="$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "=== python:    $PY ($PYVER)"
case "$PYVER" in
  3.10|3.11) ;;
  *) echo "WARNING: original runs used 3.11; 3.10 also OK. $PYVER is untested." ;;
esac

if [ ! -d "$VENV" ]; then
  echo "=== creating venv"
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --no-cache-dir --upgrade pip wheel

echo "=== installing pinned dependencies (requirements-lock.txt)"
pip install --no-cache-dir -r "$HERE/requirements-lock.txt"

echo "=== installing hydrogym (editable) from $HERE"
pip install --no-cache-dir -e "$HERE"

pip cache purge 2>/dev/null || true

# Optional: reclaim ~67 MB the experiments never touch.
# rm -rf "$HERE/docs" "$HERE/assets"

echo
echo "=== verifying JAX sees the GPU"
python - <<'PY'
import jax
print("jax", jax.__version__)
devs = jax.devices()
print("devices:", devs)
if not any(d.platform == "gpu" for d in devs):
    print("!! NO GPU VISIBLE — runs will use CPU and be very slow.")
    raise SystemExit(1)
import jax.numpy as jnp
x = jnp.ones((1024, 1024))
print("matmul check:", float((x @ x).sum()))
PY

echo
echo "=== verifying the real Kolmogorov env imports"
python -c "from hydrogym.jax.envs.kolmogorov import KolmogorovFlow; print('KolmogorovFlow OK')"

echo
echo "=== setup complete. Next:"
echo "    source $VENV/bin/activate"
echo "    cd $HERE"
echo "    nohup bash scripts/run_seed_sweep.sh 2 > logs/log_seed2_driver.log 2>&1 &"
