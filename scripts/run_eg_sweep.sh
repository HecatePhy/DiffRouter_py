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
ITERS="${ITERS:-40}"      # EG needs more outers than Adam-opt5 to diffuse+reroute

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

# ---------- 0. net index per benchmark (build if missing) ----------
# The EG global route needs a per-net corridor index. This used to be assumed present
# from the cvo campaign; on a fresh checkout it must be built, else every benchmark below
# is skipped and the sweep does nothing.
for b in $BENCH; do
  ls data/$b/net_index/*corr2*.pt >/dev/null 2>&1 && continue
  log "$b: building net index"
  $PY -u scripts/PrebuildNetIndex.py --testcase "$b" --rrg "$RRG" \
      --edge-mode directed --edge-scope corridor --corridor-width 2 \
      > "$OUT/logs/$b.netindex.log" 2>&1 || log "$b: NET INDEX FAILED (see $OUT/logs/$b.netindex.log)"
done

# ---------- 1. EG guides (one GPU each, round-robin) ----------
gpu=0
for b in $BENCH; do
  ls data/$b/net_index/*corr2*.pt >/dev/null 2>&1 || { log "$b: no net index (build failed), skip"; continue; }
  if [ ! -f "$OUT/guides/$b.eg.guide" ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY -u run_exp.py --testcase "$b" --global-only --rrg "$RRG" \
        --connectivity-solver grouped --conn-warm-start \
        --conn-col-chunk 128 --conn-cg-max-iter 8 --conn-every 5 \
        --max-iterations "$ITERS" --num-inner 5 --skip-extract \
        --init-mode shortest_path --init-off-path 0.01 \
        --optimizer-kind eg --eg-lr 0.5 --eg-clip 1.0 \
        --congestion-mode soft --congestion-tau 0.1 \
        --lam-update mult --lam-mult-eta 0.5 --lam-base 1.0 \
        --conn-sat-alpha 1.5 \
        --guide-out "$OUT/guides/$b.eg.guide" --results "$OUT/res_$b/" \
        > "$OUT/logs/$b.eg.log" 2>&1 &
    gpu=$(( (gpu+1) % NGPU ))
    [ $gpu -eq 0 ] && wait
  fi
done
wait
log "EG guides done"

# ---------- 2. Potter on the EG guide (ctrl/opt5 phys already exist in cvo) ----------
running=0
for b in $BENCH; do
  PHYS="data/$b/${b}_unrouted.phys"
  g="$OUT/guides/$b.eg.guide"; [ -f "$g" ] || continue
  o="$OUT/phys/$b.eg.phys"; [ -f "$o" ] && continue
  /usr/bin/time -v "$POTTER/build/route" -i "$PHYS" -o "$o" -d "$DEV" -t "$THREADS" -r \
      -g "$g" --guide_penalty 0.5 > "$OUT/logs/$b.eg.potter.log" 2>&1 &
  running=$((running+1))
  if [ "$running" -ge "$POT_PAR" ]; then wait; running=0; fi
done
wait
log "Potter (eg) done"

# ---------- 3. metrics for eg ----------
# potter_s from /usr/bin/time -v (if available); guide_s from run_exp's own [optimize]
# timer (the EG run is not time-wrapped, so the old "Elapsed (wall clock)" grep always
# came up empty -- that is why runtime was blank).
secs() { local t; t=$(grep -i "Elapsed (wall clock)" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9:.]+$")
         [ -z "$t" ] && { echo ""; return; }; echo "$t" | awk -F: '{s=0;for(i=1;i<=NF;i++)s=s*60+$i;printf "%.0f",s}'; }
warned_capnp=0
for b in $BENCH; do
  o="$OUT/phys/$b.eg.phys"; [ -f "$o" ] || continue
  grep -q "^$b,eg," "$CSV" && continue
  wl="$OUT/logs/$b.eg.wa.log"; tw="$OUT/logs/$b.eg.twl.log"
  [ -f "$wl" ] || (cd "$POTTER/wirelength_analyzer" && "$PY" -u wa.py "$o" > "$wl" 2>&1)
  [ -f "$tw" ] || "$PY" -u scripts/total_wirelength.py "$o" --potter "$POTTER" --quiet > "$tw" 2>&1
  cp=$(grep -iE "^Wirelength:" "$wl" 2>/dev/null | head -1 | grep -oE "[0-9]+")
  tt=$(grep -E "^[0-9]+$" "$tw" 2>/dev/null | head -1)
  # If the wirelength tools failed, say WHY (usually missing pycapnp) instead of blanking.
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
  # guide_s = global route + guide export, i.e. the whole pre-Potter flow: run_exp's
  # top-level [optimize] and [guide] timer lines (the ^ anchor skips the indented
  # per-stage [guide] prints). Does NOT include the offline wa.py/total_wirelength.py.
  opt=$(grep -oE "^\[optimize\] [0-9.]+s" "$OUT/logs/$b.eg.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  gexp=$(grep -oE "^\[guide\] [0-9.]+s" "$OUT/logs/$b.eg.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  if [ -n "$opt" ]; then gs=$(awk "BEGIN{printf \"%.0f\", $opt + ${gexp:-0}}")
  else gs=$(secs "$OUT/logs/$b.eg.log"); fi
  ps=$(secs "$OUT/logs/$b.eg.potter.log")
  echo "$b,eg,${nn:-},${gs:-},${ps:-},${cp:-},${tt:-}" >> "$CSV"
  log "$b/eg: guide_s=${gs:-?} potter_s=${ps:-?} cpwl=${cp:-?} total_wl=${tt:-?}"
done

log "DONE -> $CSV"
command -v column >/dev/null && column -s, -t "$CSV" || cat "$CSV"
