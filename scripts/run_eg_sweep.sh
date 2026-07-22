#!/bin/bash
# All-benchmark run of the EG (per-net simplex) optimizer for congestion-reducing guides.
#
# Per benchmark, self-contained: builds the net index if missing, runs the EG global route
# to produce a guide, routes it with Potter, and records {CPWL, total WL} to egsweep/.
#
# Prerequisites (checked up front, fails loudly if missing):
#   - a Python env with torch/numpy (pip install -r requirements.txt)
#   - a built Potter at $POTTER (potter/setup_potter.sh)
#   - the RRG ($RRG) and device file ($DEV)
#   - data/<name>/<name>_unrouted.phys for each benchmark
#
# To compare against the ctrl (uniform-x) and opt5 (Adam) guides, run
# scripts/run_control_vs_opt.sh first (writes cvo/results.csv); this script only produces
# the 'eg' rows. Potter runs in parallel (quality is exact regardless of contention).
#
# Config via env: BENCH, POTTER, RRG, DEV, OUT, THREADS, POT_PAR, NGPU, ITERS.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Uses whatever `python` is active. Activate your environment before running, OR set both
# CONDA_SH (path to conda.sh) and CONDA_ENV (env name) to have the script activate it.
if [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ] && [ -n "${CONDA_ENV:-}" ]; then
  # shellcheck disable=SC1090
  . "$CONDA_SH" && conda activate "$CONDA_ENV"
fi
PY=$(command -v python)
"$PY" -c "import torch" 2>/dev/null || {
  echo "ERROR: no torch in '$PY'. Activate your Python env first, or set CONDA_SH and CONDA_ENV."
  echo "       Dependencies: pip install -r requirements.txt"; exit 1; }
# The EG global route wants a GPU; a CPU-only torch runs but is far slower (and
# nvidia-smi shows nothing). Warn up front rather than per-benchmark.
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || {
  echo "WARNING: torch.cuda.is_available() is False in '$PY' -- the global route will run"
  echo "         on CPU (slow; nvidia-smi will show no usage). Your PyTorch is likely a"
  echo "         CPU-only build or its CUDA version doesn't match the driver. Reinstall the"
  echo "         matching wheel from https://pytorch.org. Continuing on CPU in 5s..."
  sleep 5; }

# Path to the built Potter (potter/setup_potter.sh builds it into third_party/Potter).
POTTER="${POTTER:-$ROOT/third_party/Potter}"
[ -x "$POTTER/build/route" ] || {
  echo "ERROR: Potter not found at '$POTTER/build/route'. Build it with potter/setup_potter.sh,"
  echo "       or set POTTER=/path/to/Potter."; exit 1; }
CVO="${CVO:-$ROOT/cvo}"
OUT="${OUT:-$ROOT/egsweep}"
RRG="${RRG:-data/rrg_xcvu3p_int.pt}"
DEV="${DEV:-data/xcvu3p.device}"
THREADS="${THREADS:-32}"
POT_PAR="${POT_PAR:-4}"
NGPU="${NGPU:-4}"
ITERS="${ITERS:-25}"      # EG converges by ~25 outers; overflow early-stop may stop sooner

mkdir -p "$OUT/logs" "$OUT/guides" "$OUT/phys"
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "benchmark,config,nets,guide_s,potter_s,cpwl,total_wl" > "$CSV"
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------- prerequisites (fail loudly, not silently) ----------
[ -f "$RRG" ] || { echo "ERROR: RRG not found at '$RRG'. Set RRG=/path/to/rrg_*.pt."; exit 1; }
[ -f "$DEV" ] || { echo "ERROR: device file not found at '$DEV'. Set DEV=/path/to/*.device."; exit 1; }

if [ -z "${BENCH:-}" ]; then
  BENCH=$(for d in data/*/; do b=$(basename "$d"); f="$d${b}_unrouted.phys"
          [ -f "$f" ] && echo "$(stat -c%s "$f") $b"; done | sort -n | awk '{print $2}')
fi
if [ -z "$BENCH" ]; then
  echo "ERROR: no benchmarks found. Expected data/<name>/<name>_unrouted.phys for each"
  echo "       benchmark (populate data/ first), or pass BENCH='name1 name2 ...'."; exit 1; fi
log "benchmarks: $(echo $BENCH | wc -w) [$(echo $BENCH | tr '\n' ' ')]"

# seconds from a /usr/bin/time -v log
secs() { local t; t=$(grep -i "Elapsed (wall clock)" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9:.]+$")
         [ -z "$t" ] && { echo ""; return; }; echo "$t" | awk -F: '{s=0;for(i=1;i<=NF;i++)s=s*60+$i;printf "%.0f",s}'; }

# ---------- per-benchmark pipeline: net index -> EG route -> Potter -> metrics ----------
# One benchmark start-to-finish before the next, so each result row lands as it completes.
# GPU is fixed per run ($GPU, default 0). For cross-benchmark parallelism, launch several
# instances with disjoint BENCH sets on different GPUs.
GPU="${GPU:-0}"
ntot=$(echo $BENCH | wc -w); i=0
warned_capnp=0
for b in $BENCH; do
  i=$((i+1))
  log "==================== ($i/$ntot) $b ===================="

  # 1. net index (build if missing)
  if ! ls data/$b/net_index/*corr2*.pt >/dev/null 2>&1; then
    log "  [1/4] $b: building net index -> $OUT/logs/$b.netindex.log"
    $PY -u scripts/PrebuildNetIndex.py --testcase "$b" --rrg "$RRG" \
        --edge-mode directed --edge-scope corridor --corridor-width 2 \
        > "$OUT/logs/$b.netindex.log" 2>&1 \
      || { log "  $b: NET INDEX FAILED (see $OUT/logs/$b.netindex.log), skipping"; continue; }
  else
    log "  [1/4] $b: net index present"
  fi

  # 2. EG global route -> guide
  g="$OUT/guides/$b.eg.guide"
  if [ -f "$g" ]; then
    log "  [2/4] $b: guide exists, skip global route"
  else
    log "  [2/4] $b: EG global route (GPU $GPU, up to $ITERS iters) -> tail -f $OUT/logs/$b.eg.log"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u run_exp.py --testcase "$b" --global-only --rrg "$RRG" \
        --connectivity-solver grouped --conn-warm-start \
        --conn-col-chunk 128 --conn-cg-max-iter 8 --conn-every 5 \
        --max-iterations "$ITERS" --num-inner 5 --skip-extract \
        --init-mode shortest_path --init-off-path 0.01 \
        --optimizer-kind eg --eg-lr 0.5 --eg-clip 1.0 \
        --congestion-mode soft --congestion-tau 0.1 \
        --lam-update mult --lam-mult-eta 0.5 --lam-base 1.0 \
        --conn-sat-alpha 1.5 --overflow-stop-frac 0.01 \
        --guide-out "$g" --results "$OUT/res_$b/" \
        > "$OUT/logs/$b.eg.log" 2>&1 \
      || { log "  $b: GLOBAL ROUTE FAILED (see $OUT/logs/$b.eg.log), skipping"; continue; }
    gso=$(grep -oE "^\[optimize\] [0-9.]+s" "$OUT/logs/$b.eg.log" | tail -1 | grep -oE "[0-9.]+")
    log "  [2/4] $b: guide written ($(wc -l < "$g" 2>/dev/null) nets, global route ${gso%.*}s)"
  fi

  # 3. Potter detailed route
  o="$OUT/phys/$b.eg.phys"
  if [ -f "$o" ]; then
    log "  [3/4] $b: phys exists, skip Potter"
  else
    log "  [3/4] $b: Potter ($THREADS threads) -> tail -f $OUT/logs/$b.eg.potter.log"
    /usr/bin/time -v "$POTTER/build/route" -i "data/$b/${b}_unrouted.phys" -o "$o" \
        -d "$DEV" -t "$THREADS" -r -g "$g" --guide_penalty 0.5 \
        > "$OUT/logs/$b.eg.potter.log" 2>&1 \
      || { log "  $b: POTTER FAILED (see $OUT/logs/$b.eg.potter.log), skipping"; continue; }
    _ps=$(secs "$OUT/logs/$b.eg.potter.log"); log "  [3/4] $b: routed (${_ps:-?}s)"
  fi

  # 4. metrics (CPWL + total WL)
  grep -q "^$b,eg," "$CSV" && { log "  [4/4] $b: already in CSV"; continue; }
  log "  [4/4] $b: computing CPWL + total WL"
  wl="$OUT/logs/$b.eg.wa.log"; tw="$OUT/logs/$b.eg.twl.log"
  [ -f "$wl" ] || (cd "$POTTER/wirelength_analyzer" && "$PY" -u wa.py "$o" > "$wl" 2>&1)
  [ -f "$tw" ] || "$PY" -u scripts/total_wirelength.py "$o" --potter "$POTTER" --quiet > "$tw" 2>&1
  cp=$(grep -iE "^Wirelength:" "$wl" 2>/dev/null | head -1 | grep -oE "[0-9]+")
  tt=$(grep -E "^[0-9]+$" "$tw" 2>/dev/null | head -1)
  if [ -z "$cp$tt" ] && [ "$warned_capnp" = 0 ]; then
    if grep -qiE "ModuleNotFoundError|ImportError|No module named|capnp" "$wl" "$tw" 2>/dev/null; then
      echo "WARNING: wa.py / total_wirelength.py failed to import (cpwl/total_wl will be"
      echo "         blank). Potter's wirelength analyzer needs pycapnp: pip install pycapnp"
      echo "         (see $wl and $tw for the exact error)."
    else
      echo "WARNING: no cpwl/total_wl for $b -- see $wl and $tw."
    fi
    warned_capnp=1
  fi
  nn=$(grep -oE "[0-9]+ nets matched" "$OUT/logs/$b.eg.potter.log" 2>/dev/null | head -1 | grep -oE "^[0-9]+")
  # guide_s = global route + guide export ([optimize] + [guide] timers; not the analysis).
  opt=$(grep -oE "^\[optimize\] [0-9.]+s" "$OUT/logs/$b.eg.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  gexp=$(grep -oE "^\[guide\] [0-9.]+s" "$OUT/logs/$b.eg.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  if [ -n "$opt" ]; then gs=$(awk "BEGIN{printf \"%.0f\", $opt + ${gexp:-0}}")
  else gs=$(secs "$OUT/logs/$b.eg.log"); fi
  ps=$(secs "$OUT/logs/$b.eg.potter.log")
  echo "$b,eg,${nn:-},${gs:-},${ps:-},${cp:-},${tt:-}" >> "$CSV"
  log "  DONE $b: guide_s=${gs:-?}s potter_s=${ps:-?}s cpwl=${cp:-?} total_wl=${tt:-?}"
done

log "ALL DONE -> $CSV"
command -v column >/dev/null && column -s, -t "$CSV" || cat "$CSV"
