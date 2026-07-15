#!/bin/bash
# End-to-end DiffRouter -> Potter campaign over benchmarks, both flow modes.
#
# Per benchmark:  net index -> global route -> guide -> Potter (guided + unguided) -> CPWL
# Results appended to $OUT/results.csv as they complete (safe to interrupt/resume:
# stages are skipped when their outputs already exist).
#
# Usage:
#   scripts/run_all_benchmarks.sh                 # all benchmarks, both modes
#   BENCH="logicnets_jscl vtr_mcml" scripts/run_all_benchmarks.sh
#   MODES="runtime" THREADS=64 scripts/run_all_benchmarks.sh
#
# Env: BENCH, MODES(cpwl|runtime|both), THREADS, ITERS, NGPU, POTTER, OUT, SKIP_UNGUIDED
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Use the project env's python (the system python is too old for this code).
CONDA_ENV="${CONDA_ENV:-diffrouter}"
if [ -z "${CONDA_SH:-}" ]; then
  for c in "$HOME/miniforge3" "$(dirname "$(dirname "$(command -v conda 2>/dev/null || echo /nonexistent)")")" \
           /home/usr1/"$USER"/miniforge3 "$HOME/miniconda3" "$HOME/anaconda3"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && { CONDA_SH="$c/etc/profile.d/conda.sh"; break; }
  done
fi
if [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  . "$CONDA_SH" && conda activate "$CONDA_ENV"
fi
python -c "import torch" 2>/dev/null || {
  echo "ERROR: no torch in '$(command -v python)'. Set CONDA_SH=/path/to/conda.sh CONDA_ENV=<env>."; exit 1; }

POTTER="${POTTER:-/tmp/claude-5033/-home-usr1-xg3787-projs-DiffRouter-py/4dd12ae8-e3ff-4789-826a-5238420035fd/scratchpad/Potter}"
OUT="${OUT:-$ROOT/campaign}"
THREADS="${THREADS:-32}"
ITERS="${ITERS:-60}"
NGPU="${NGPU:-4}"
MODES="${MODES:-both}"
RRG="${RRG:-data/rrg_xcvu3p_int.pt}"
DEV="${DEV:-data/xcvu3p.device}"
SKIP_UNGUIDED="${SKIP_UNGUIDED:-0}"

# default: every benchmark that has an unrouted .phys, smallest first (fail fast)
if [ -z "${BENCH:-}" ]; then
  BENCH=$(for d in data/*/; do
            b=$(basename "$d"); f="$d${b}_unrouted.phys"
            [ -f "$f" ] && echo "$(stat -c%s "$f") $b"
          done | sort -n | awk '{print $2}')
fi

mkdir -p "$OUT/logs"
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "benchmark,mode,stage_global_s,stage_guide_s,stage_potter_s,total_s,cpwl,nets,notes" > "$CSV"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
secs() { # extract "Elapsed (wall clock)" m:ss / h:mm:ss -> seconds
  local t; t=$(grep -i "Elapsed (wall clock)" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9:.]+$")
  [ -z "$t" ] && { echo ""; return; }
  echo "$t" | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; printf "%.0f", s}'
}
cpwl() { grep -iE "^Wirelength:" "$1" 2>/dev/null | head -1 | grep -oE "[0-9]+"; }

run_mode() { # benchmark mode
  local b=$1 mode=$2
  local res="$OUT/$b/$mode"
  mkdir -p "$res"
  local guide="$OUT/$b/$b.$mode.guide"
  local gl="$OUT/logs/$b.$mode.global.log" ge="$OUT/logs/$b.$mode.guide.log" pl="$OUT/logs/$b.$mode.potter.log"
  local t_g t_e t_p

  # ---- 1. global route
  if [ ! -f "$res/$b/checkpoint/global_x.pt" ]; then
    log "$b/$mode: global route"
    local extra="--conn-every 5"
    [ "$mode" = "runtime" ] && extra="--conn-every 5 --conn-super-sink --conn-multi-gpu $NGPU"
    /usr/bin/time -v python -u run_exp.py --testcase "$b" --global-only \
        --rrg "$RRG" --connectivity-solver grouped --conn-warm-start $extra \
        --max-iterations "$ITERS" --num-inner 5 --skip-extract \
        --guide-out "$guide" --results "$res/" > "$gl" 2>&1
  fi
  t_g=$(secs "$gl")
  if [ ! -f "$res/$b/checkpoint/global_x.pt" ]; then
    echo "$b,$mode,,,,,,,GLOBAL_FAILED" >> "$CSV"; log "$b/$mode: global FAILED (see $gl)"; return 1
  fi

  # ---- 2. guide: written INLINE by the global-route step above (GPU batched
  # Bellman-Ford on the in-memory router). Fall back to the standalone exporter if
  # the guide is missing (e.g. the global route was cached from an older run).
  if [ ! -f "$guide" ]; then
    log "$b/$mode: guide export (GPU, standalone)"
    /usr/bin/time -v python -u scripts/gpu_guide_export.py \
        --x "$res/$b/checkpoint/global_x.pt" --rrg "$RRG" --testcase "$b" \
        --out "$guide" > "$ge" 2>&1
    t_e=$(secs "$ge")
  else
    t_e=0   # produced inline by the global-route stage
  fi
  [ -f "$guide" ] || { echo "$b,$mode,$t_g,,,,,,GUIDE_FAILED" >> "$CSV"; log "$b/$mode: guide FAILED"; return 1; }

  # ---- 3. Potter (guided). isolated: one route at a time
  local pen=2; [ "$mode" = "runtime" ] && pen=0.5
  local phys="data/$b/${b}_unrouted.phys" outphys="$OUT/$b/${b}.$mode.routed.phys"
  if [ ! -f "$outphys" ]; then
    log "$b/$mode: Potter -t $THREADS -r --guide_penalty $pen"
    /usr/bin/time -v "$POTTER/build/route" -i "$phys" -o "$outphys" -d "$DEV" \
        -t "$THREADS" -r -g "$guide" --guide_penalty $pen > "$pl" 2>&1
  fi
  t_p=$(secs "$pl")
  [ -f "$outphys" ] || { echo "$b,$mode,$t_g,$t_e,,,,,POTTER_FAILED" >> "$CSV"; log "$b/$mode: Potter FAILED"; return 1; }

  # ---- 4. CPWL
  local wl="$OUT/logs/$b.$mode.wa.log"
  [ -f "$wl" ] || (cd "$POTTER/wirelength_analyzer" && python -u wa.py "$outphys" > "$wl" 2>&1)
  local nets; nets=$(grep -oE "[0-9]+ nets matched" "$pl" | head -1 | grep -oE "^[0-9]+")
  local total=$(( ${t_g:-0} + ${t_e:-0} + ${t_p:-0} ))
  echo "$b,$mode,${t_g:-},${t_e:-},${t_p:-},$total,$(cpwl "$wl"),${nets:-},ok" >> "$CSV"
  log "$b/$mode: total=${total}s cpwl=$(cpwl "$wl")"
}

for b in $BENCH; do
  phys="data/$b/${b}_unrouted.phys"
  [ -f "$phys" ] || { log "$b: no .phys, skip"; continue; }
  mkdir -p "$OUT/$b"

  # ---- net index (once per benchmark; C++ extractor, no Java)
  if ! ls data/$b/net_index/*corr2*.pt >/dev/null 2>&1; then
    log "$b: prebuild net index"
    python -u scripts/PrebuildNetIndex.py --testcase "$b" --rrg "$RRG" \
        --edge-mode directed --edge-scope corridor --corridor-width 2 \
        > "$OUT/logs/$b.netindex.log" 2>&1 \
      || { log "$b: net index FAILED"; echo "$b,,,,,,,,NETINDEX_FAILED" >> "$CSV"; continue; }
  fi

  # ---- unguided Potter reference (once per benchmark)
  if [ "$SKIP_UNGUIDED" != "1" ] && [ ! -f "$OUT/$b/${b}.unguided.routed.phys" ]; then
    log "$b/unguided: Potter reference"
    ul="$OUT/logs/$b.unguided.potter.log"
    /usr/bin/time -v "$POTTER/build/route" -i "$phys" -o "$OUT/$b/${b}.unguided.routed.phys" \
        -d "$DEV" -t "$THREADS" -r > "$ul" 2>&1
    uw="$OUT/logs/$b.unguided.wa.log"
    [ -f "$OUT/$b/${b}.unguided.routed.phys" ] && \
      (cd "$POTTER/wirelength_analyzer" && python -u wa.py "$OUT/$b/${b}.unguided.routed.phys" > "$uw" 2>&1)
    tp=$(secs "$ul"); echo "$b,unguided,0,0,${tp:-},${tp:-},$(cpwl "$uw"),,ok" >> "$CSV"
    log "$b/unguided: total=${tp}s cpwl=$(cpwl "$uw")"
  fi

  case "$MODES" in
    both)    run_mode "$b" cpwl; run_mode "$b" runtime ;;
    cpwl)    run_mode "$b" cpwl ;;
    runtime) run_mode "$b" runtime ;;
  esac
done

log "CAMPAIGN DONE -> $CSV"
column -s, -t "$CSV"
