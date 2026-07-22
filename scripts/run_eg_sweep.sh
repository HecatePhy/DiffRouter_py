#!/bin/bash
# All-benchmark test of the EG (per-net simplex) optimizer for congestion-reducing guides.
#
# Per benchmark, produces an 'eg' guide alongside the existing ctrl/opt5 guides from the
# cvo campaign, routes all with Potter, and reports {extracted overflow, CPWL, total WL}.
#
#   ctrl : uniform-x guide (zero optimization)         -- corridor structure only
#   opt5 : 5-iter Adam guide (box)                     -- measured ~ ctrl (no rerouting)
#   eg   : EG guide (per-net simplex, reroutes)        -- the new mechanism
#
# eg vs ctrl on total WL / CPWL is the question that matters. On boom_soc_v2 the EG guide
# cut EXTRACTED overflow -41% vs -0.12% for Adam; this checks whether that (a) generalizes
# and (b) reaches the Potter-level metrics.
#
# Reuses cvo/ net indices, ctrl/opt5 guides, and phys. Potter runs in parallel (quality is
# exact regardless of contention; runtimes are indicative only).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONDA_SH="${CONDA_SH:-/home/usr1/xg3787/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] && { . "$CONDA_SH"; conda activate "${CONDA_ENV:-diffrouter}"; }
PY=$(which python)
$PY -c "import torch" 2>/dev/null || { echo "ERROR: no torch"; exit 1; }

POTTER="${POTTER:-/tmp/claude-5033/-home-usr1-xg3787-projs-DiffRouter-py/4dd12ae8-e3ff-4789-826a-5238420035fd/scratchpad/Potter}"
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

if [ -z "${BENCH:-}" ]; then
  BENCH=$(for d in data/*/; do b=$(basename "$d"); f="$d${b}_unrouted.phys"
          [ -f "$f" ] && echo "$(stat -c%s "$f") $b"; done | sort -n | awk '{print $2}')
fi

# ---------- 1. EG guides (one GPU each, round-robin) ----------
gpu=0
for b in $BENCH; do
  ls data/$b/net_index/*corr2*.pt >/dev/null 2>&1 || { log "$b: no net index, skip"; continue; }
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

# ---------- 3. metrics for eg (and copy ctrl/opt5 from cvo for side-by-side) ----------
secs() { local t; t=$(grep -i "Elapsed (wall clock)" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9:.]+$")
         [ -z "$t" ] && { echo ""; return; }; echo "$t" | awk -F: '{s=0;for(i=1;i<=NF;i++)s=s*60+$i;printf "%.0f",s}'; }
for b in $BENCH; do
  o="$OUT/phys/$b.eg.phys"; [ -f "$o" ] || continue
  grep -q "^$b,eg," "$CSV" && continue
  wl="$OUT/logs/$b.eg.wa.log"; tw="$OUT/logs/$b.eg.twl.log"
  [ -f "$wl" ] || (cd "$POTTER/wirelength_analyzer" && $PY -u wa.py "$o" > "$wl" 2>&1)
  [ -f "$tw" ] || $PY -u scripts/total_wirelength.py "$o" --potter "$POTTER" --quiet > "$tw" 2>&1
  cp=$(grep -iE "^Wirelength:" "$wl" 2>/dev/null | head -1 | grep -oE "[0-9]+")
  tt=$(grep -E "^[0-9]+$" "$tw" 2>/dev/null | head -1)
  nn=$(grep -oE "[0-9]+ nets matched" "$OUT/logs/$b.eg.potter.log" 2>/dev/null | head -1 | grep -oE "^[0-9]+")
  gs=$(secs "$OUT/logs/$b.eg.log")
  echo "$b,eg,${nn:-},${gs:-},$(secs "$OUT/logs/$b.eg.potter.log"),${cp:-},${tt:-}" >> "$CSV"
  log "$b/eg: cpwl=${cp:-?} total_wl=${tt:-?}"
done

log "DONE -> $CSV"
column -s, -t "$CSV"
