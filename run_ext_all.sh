#!/usr/bin/env bash
# =============================================================================
#  HydroGym — ek çalışmalar (IYILESTIRME_ONERILERI.md paketleri A + B + C)
#  TEK KOMUT:   bash run_ext_all.sh
#
#  - venv yoksa kurar (setup_ext_env.sh), sonra üç paketi sırayla koşar.
#  - Her config kendi JSON dosyasına yazılır; dosya varsa ATLANIR (resume).
#  - Tek GPU, hepsi seri. nohup ile arka planda bırakılabilir:
#        nohup bash run_ext_all.sh > log_ext_all.log 2>&1 &
#  - Bitince analyze_ext.py çalışır -> EXT_SONUCLAR.md üretir.
#
#  Paket / seed / Re listesini ortam değişkeniyle daraltabilirsin:
#        EPISODES=300 SEEDS_A="0 1" PACKAGES="A C" bash run_ext_all.sh
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---- ayarlanabilir parametreler --------------------------------------------
EPISODES="${EPISODES:-500}"
PACKAGES="${PACKAGES:-A B C}"          # hangi paketler koşulacak
RE_LIST="${RE_LIST:-100 150 200 250}"  # paket A: Reynolds taraması
SEEDS_A="${SEEDS_A:-0 1 2}"            # paket A: seed sayısı (memo: 3)
SEEDS_B="${SEEDS_B:-0 1 2 3 4}"        # paket B: güçlü PPO
SEEDS_C="${SEEDS_C:-0 1 2 3 4}"        # paket C: GSAC korumaları
PPO_EPOCHS="${PPO_EPOCHS:-4}"          # paket B: PPO iç-döngü epoch sayısı (1 = eski davranış)
PPO_MINIB="${PPO_MINIB:-1}"            # paket B: minibatch sayısı (batch = steps = 15)

OUT="$HERE/results_ext"
mkdir -p "$OUT"

# ---- venv --------------------------------------------------------------------
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PY="python3"
elif [ -x "$HERE/venv/bin/python3" ]; then
  PY="$HERE/venv/bin/python3"
elif [ -x "$HERE/../venv/bin/python3" ]; then
  PY="$HERE/../venv/bin/python3"
else
  echo ">>> venv bulunamadı — kuruluyor (setup_ext_env.sh, ~10-15 dk, ağ gerekir)"
  bash "$HERE/setup_ext_env.sh" || { echo "!! kurulum başarısız"; exit 1; }
  PY="$HERE/venv/bin/python3"
fi
echo ">>> python: $PY"
"$PY" -c "import jax; assert any(d.platform=='gpu' for d in jax.devices()), 'GPU yok'; print('jax', jax.__version__, jax.devices())" \
  || { echo "!! JAX GPU görmüyor — CPU çok yavaş, iptal"; exit 1; }

export PYTHONPATH="$HERE:${PYTHONPATH:-}"    # gömülü hydrogym paketi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export PYTHONUNBUFFERED=1

run() {   # run <script> <out.json> <arg...>
  local script="$1"; local out="$2"; shift 2
  if [ -f "$out" ]; then
    echo "=== ATLA (var): $(basename "$out")"
    return 0
  fi
  echo "=== BAŞLA $(basename "$out")  $(date '+%H:%M:%S')"
  "$PY" -u "$script" "$@" --out "$out" > "${out%.json}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! HATA ($rc): $(basename "$out") — log: ${out%.json}.log"
    mv "$out" "${out%.json}.FAILED.json" 2>/dev/null || true
  else
    echo "=== BİTTİ  $(basename "$out")  $(date '+%H:%M:%S')"
  fi
}

has_pkg() { [[ " $PACKAGES " == *" $1 "* ]]; }

# ---------------------------------------------------------------------------
# PAKET A — Reynolds taraması  {100,150,200,250} x {ppo,gppo,sac,gsac} x seed
# ---------------------------------------------------------------------------
if has_pkg A; then
  echo; echo "############ PAKET A — Reynolds taraması ############"
  for RE in $RE_LIST; do
    for SEED in $SEEDS_A; do
      run run_real_experiment.py "$OUT/A_ppo_re${RE}_seed${SEED}.json" \
          --episodes "$EPISODES" --seed "$SEED" --only ppo_high  --re "$RE"
      run run_real_experiment.py "$OUT/A_gppo_re${RE}_seed${SEED}.json" \
          --episodes "$EPISODES" --seed "$SEED" --only gppo_high --re "$RE"
      run run_sac_experiment.py  "$OUT/A_sac_re${RE}_seed${SEED}.json" \
          --episodes "$EPISODES" --seed "$SEED" --only sac_high  --re "$RE"
      run run_sac_experiment.py  "$OUT/A_gsac_re${RE}_seed${SEED}.json" \
          --episodes "$EPISODES" --seed "$SEED" --only gsac_high --re "$RE"
    done
  done
fi

# ---------------------------------------------------------------------------
# PAKET B — güçlü PPO baseline
#   Epizot başına tek tam-batch adım yerine gerçek PPO iç-döngüsü:
#   PPO_EPOCHS geçiş x PPO_MINIB minibatch, PPO clip'i artık fiilen devrede.
#   (Avantaj normalizasyonu zaten ppo_loss içinde her zaman açık.)
#   ppo_high / gppo_high @ Re=300, 5 seed.
# ---------------------------------------------------------------------------
if has_pkg B; then
  echo; echo "############ PAKET B — güçlü PPO baseline ############"
  for SEED in $SEEDS_B; do
    run run_real_experiment.py "$OUT/B_ppo_strong_seed${SEED}.json" \
        --episodes "$EPISODES" --seed "$SEED" --only ppo_high \
        --ppo-epochs "$PPO_EPOCHS" --ppo-minibatches "$PPO_MINIB"
    run run_real_experiment.py "$OUT/B_gppo_strong_seed${SEED}.json" \
        --episodes "$EPISODES" --seed "$SEED" --only gppo_high \
        --ppo-epochs "$PPO_EPOCHS" --ppo-minibatches "$PPO_MINIB"
  done
fi

# ---------------------------------------------------------------------------
# PAKET C — GSAC-taban korumaları  (gsac_high @ Re=300, 5 seed, 4 varyant)
#   clip      : analitik-gradyan global-norm kırpma (eşik 15)
#   fallback  : non-finite analitik gradyan -> o epizot saf SAC adımı
#   warmup    : ilk 25 epizot analitik enjeksiyon yok
#   adaptive  : GI-PPO alpha tarzı — norm 20'yi aşınca analitik ağırlığı küçült
#   combo     : clip + fallback + warmup + adaptive hepsi birden
# ---------------------------------------------------------------------------
if has_pkg C; then
  echo; echo "############ PAKET C — GSAC korumaları ############"
  for SEED in $SEEDS_C; do
    run run_sac_experiment.py "$OUT/C_gsac_clip_seed${SEED}.json" \
        --episodes "$EPISODES" --seed "$SEED" --only gsac_high --grad-clip 15
    run run_sac_experiment.py "$OUT/C_gsac_fallback_seed${SEED}.json" \
        --episodes "$EPISODES" --seed "$SEED" --only gsac_high --nonfinite-fallback
    run run_sac_experiment.py "$OUT/C_gsac_warmup_seed${SEED}.json" \
        --episodes "$EPISODES" --seed "$SEED" --only gsac_high --analytic-warmup 25
    run run_sac_experiment.py "$OUT/C_gsac_adaptive_seed${SEED}.json" \
        --episodes "$EPISODES" --seed "$SEED" --only gsac_high --adaptive-mix --adaptive-target 20
    run run_sac_experiment.py "$OUT/C_gsac_combo_seed${SEED}.json" \
        --episodes "$EPISODES" --seed "$SEED" --only gsac_high \
        --grad-clip 15 --nonfinite-fallback --analytic-warmup 25 --adaptive-mix --adaptive-target 20
  done
fi

echo; echo "############ ANALİZ ############"
"$PY" analyze_ext.py --dir "$OUT" --out "$HERE/EXT_SONUCLAR.md" || true
echo
echo ">>> HEPSİ BİTTİ  $(date)"
echo ">>> Sonuç tablosu: $HERE/EXT_SONUCLAR.md"
echo ">>> Ham JSON'lar : $OUT/"
