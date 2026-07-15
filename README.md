# DiffRouter-py

Differentiable global router for FPGA routing (FPGA24 Routing Contest). Uses PyTorch for optimization with Augmented Lagrangian.

Optionally emits **GR route guides** for [Potter](https://github.com/diriLin/Potter) (parallel
detailed router) — see [Flow Modes](#flow-modes-diffrouter--potter).

---

## Flow Modes (DiffRouter → Potter)

The global route can be exported as a **GR route guide** for Potter, which routes each
net preferring its guide corridor.

### Setting up Potter (one-time)

Potter is a separate BSD-3 project. We vendor only our patch (which adds `-g/--guide`
and `--guide_penalty`), not their source, so it stays easy to track upstream:

```bash
potter/setup_potter.sh                 # clone at pinned commit + apply patch + build
ln -sf "$PWD/data/xcvu3p.device" third_party/Potter/xcvu3p.device
```

This produces `third_party/Potter/build/route`. Needs cmake, a C++17 compiler, zlib and
boost-serialization (Cap'n Proto is bundled with Potter). The patch itself is
`potter/0001-diffrouter-gr-guidance.patch` — it adds the `-g/--guide` and
`--guide_penalty` options, a soft out-of-guide term in the A* node cost, and net-name
capture so guides can be matched to nets.

### The two modes

| mode | global route | Potter penalty | optimizes for |
|------|-------------|----------------|---------------|
| **(1) cpwl-driven** | max-connectivity (`--conn-every 5`) | 2 (strong) | critical-path wirelength |
| **(2) runtime-driven** | fastest (`+ --conn-super-sink --conn-multi-gpu`) | 0.5 (gentle) | end-to-end runtime |

Both modes use the same **tight per-net shortest-path guide** (written inline by
`--guide-out`); they differ in the global-route config and `--guide_penalty`. Guide
*tightness* is what delivers the benefit — a loose guide is equivalent to no guide (see
[Guide export](#guide-export-for-potter)).

### (1) cpwl-driven — best wirelength

Max-connectivity global route → tight Dijkstra path guide → strong guide penalty.

```bash
# 1. global route + guide (guide is written inline from the in-memory router)
python run_exp.py --testcase boom_soc_v2 --global-only \
    --connectivity-solver grouped --conn-warm-start \
    --conn-col-chunk 128 --conn-cg-max-iter 8 --conn-every 5 \
    --max-iterations 60 --num-inner 5 \
    --guide-out bsv2.guide --skip-extract

# 2. Potter with strong guide adherence
third_party/Potter/build/route -i data/boom_soc_v2/boom_soc_v2_unrouted.phys \
    -o routed.phys -d data/xcvu3p.device -t 32 -r \
    -g bsv2.guide --guide_penalty 2
```

### (2) runtime-driven — fastest flow

Fastest global route → same tight Dijkstra guide → gentle penalty.

```bash
# 1. global route + guide: conn-every + super-sink + multi-GPU
python run_exp.py --testcase boom_soc_v2 --global-only \
    --connectivity-solver grouped --conn-warm-start \
    --conn-col-chunk 128 --conn-cg-max-iter 8 \
    --conn-every 5 --conn-super-sink --conn-multi-gpu 4 \
    --max-iterations 60 --num-inner 5 \
    --guide-out bsv2.guide --skip-extract

# 2. Potter, gentle guide (nudge, not force)
third_party/Potter/build/route -i data/boom_soc_v2/boom_soc_v2_unrouted.phys \
    -o routed.phys -d data/xcvu3p.device -t 32 -r \
    -g bsv2.guide --guide_penalty 0.5
```

`--guide-out` writes the guide **inline**, reusing the router already in memory — no
save/reload round-trip. `--skip-extract` skips the tile-path extraction, which the guide
flow does not need. (`scripts/gpu_guide_export.py` does the same from a saved
`global_x.pt`, if you want the guide after the fact.)

### Guide penalty = the mode dial

`--guide_penalty` is a **soft** out-of-guide cost added to Potter's A\* node cost
(TritonRoute/CUGR-style; soft so a net can still escape a locally-infeasible guide).
It is the main dial between the two modes:

| penalty | behaviour |
|---------|-----------|
| ~0.5 | gentle nudge — fastest routing, most of the wirelength gain |
| ~1 | trade-off |
| ~2+ | strong adherence — best wirelength, slower routing (routes forced into corridors) |


---

## Default Setup (recommended)

DiffRouter targets **FPGA24-style partial routing**: pre-placed designs with some nets already routed in `sources` and remaining work in `stubs` ([contest inputs](https://xilinx.github.io/fpga24_routing_contest/start.html)).

### RRG: INT-only tile graph (C++ Interchange extract)

| Property | Value |
|----------|-------|
| **Nodes** | One per `INT` tile (not `INT_INTF_*`) |
| **Edges** | Undirected tile hops at fabric H/V distances `{1, 2, 4, 12}` |
| **Capacity** | Aggregated GENERAL routing resources per tile pair |
| **Wirelength** | FPGA24 contest PIP scores (`edge_wl_score`) |
| **Default file** | `data/rrg_xcvu3p_int.pt` |
| **Extractor** | `cpp/build/extract_rrg` via `scripts/ExtractRRG.py --extractor cpp` |

Validate after extract (expect edges only at allowed Manhattan distances):

```bash
python scripts/analyze_rrg_distances.py data/rrg_xcvu3p_int.pt
```

**Not used for global routing:** `rrg_xcvu3p_int_adj.pt` (adjacent-only, no long lines) and RapidWright wire-closure extract. See [Legacy](#legacy--debugging) below.

### Nets to route: Potter-aligned stub rule

From each `*_unrouted.phys` (FPGA Interchange physical netlist):

| Condition | Action |
|-----------|--------|
| `type == gnd` or `type == vcc` | **Skip** (power tiles) |
| `type == signal` and `stubs` empty | **Skip** (fully routed, including most clocks) |
| `type == signal` and `stubs` non-empty | **Route** (partially unrouted) |

Implemented in `cpp/build/extract_net_index` and used by default in `scripts/PrebuildNetIndex.py` (without `--use-java`).

Net index cache path (auto):  
`data/<testcase>/net_index/rrg_xcvu3p_int_<edge_mode>_stubs_mf<min_fanout>_exp10.pt`

> **The net index is a one-time, per-testcase local cost that every later run reuses.**
> Build it once with `scripts/PrebuildNetIndex.py`; `run_exp.py` (and the campaign script)
> auto-detect the cache and skip rebuilding when it exists. The first `GlobalRouter.load`
> additionally writes a **compiled cache** (`<net_index>.<edge_mode>.compiled.pt`) of the
> built flat tensors, so subsequent loads skip both the net-index deserialize and the
> flat-array build. Neither artifact is shipped in the repo — they are large (GBs per
> design) and machine-local, so `data/` and `*.pt` are gitignored. Regenerate locally;
> the C++ extractor needs no Java and is fast.

### One-time + per-testcase workflow

```bash
make -C cpp build

# Device RRG (once)
python scripts/ExtractRRG.py --extractor cpp --device-file data/xcvu3p.device \
  -o data/rrg_xcvu3p_int.pt
python scripts/analyze_rrg_distances.py data/rrg_xcvu3p_int.pt

# Net index (once per testcase)
python scripts/PrebuildNetIndex.py --testcase boom_soc_v2 --rrg data/rrg_xcvu3p_int.pt

# Run
python run_exp.py --testcase boom_soc_v2 --global-only
```

---

## End-to-End Pipeline (`run_exp.py`)

Full pipeline: global optimize → tile path extraction → detailed route → write `.phys`/`.dcp`.

```bash
# Full end-to-end (global + detailed + output)
python run_exp.py --testcase boom_med_pb --max-iterations 200 --quiet

# Global routing only (saves results/<testcase>/checkpoint/global_x.pt)
python run_exp.py --testcase boom_soc_v2 --global-only --max-iterations 500 --quiet

# Resume / skip global if checkpoint exists
python run_exp.py --testcase boom_soc_v2 --skip-global

# With congestion GIF during global opt
python run_exp.py --testcase boom_med_pb --viz --max-iterations 200
```

| Option | Default | Description |
|--------|---------|-------------|
| `--testcase` | boom_soc_v2 | Design name under `data/` |
| `--rrg` | data/rrg_xcvu3p_int.pt | Pre-extracted RRG (C++ Interchange extract) |
| `--max-iterations` | 1000 | Global AL iterations |
| `--conn-net-batch` | 0 | Legacy: nets per iter for connectivity (0=all; ignored by `cg`/`grouped`) |
| `--w-flow` | 1.0 | Flow conservation soft penalty weight |
| `--connectivity-solver` | cg | `solve`, `cg`, or `grouped` (recommended — see below) |
| `--global-only` | - | Skip detailed routing |
| `--skip-global` | - | Load cached `global_x.pt` |
| `--route-threshold` | 0.01 | Edge threshold for path extraction |
| `--max-nets` | - | Limit nets (quick tests) |
| `--edge-mode` | directed | `directed` or `undirected` (undirected: ~½ vars, no flow term) |
| `--net-index` | auto path | Pre-built net index `.pt` (required unless `--live-build-nets`) |
| `--live-build-nets` | - | Debug: build net list from Java design |
| `--resume` | - | Checkpoint path for global opt |

### Connectivity solver + runtime options

The connectivity (effective-resistance) term dominates global-route runtime. Its grouped
CG is **kernel-launch-bound**, so the effective levers reduce the *number of evaluations*
and *columns* rather than arithmetic cost.

| Option | Default | Description |
|--------|---------|-------------|
| `--connectivity-solver` | cg | `grouped` = net-grouped subgraph CG (each chunk solves only its nets' subgraph). Far faster and lower-memory than `cg`/`solve` at scale — **recommended** |
| `--conn-warm-start` | on | Initialize CG from the previous iteration's solution (`x` moves slowly, so few iterations suffice) |
| `--conn-every K` | 1 | Evaluate connectivity every K inner iters; other steps optimize wirelength/congestion/flow only. Large speedup, connectivity preserved |
| `--conn-super-sink` | off | One connectivity column per net (source → merged super-sink) instead of one per sink. Much faster and yields a far less congested route; slightly lower sink-pair connectivity |
| `--conn-multi-gpu N` | 1 | Shard connectivity groups across N GPUs (bit-exact vs single-GPU) |
| `--conn-cg-max-iter` | 100 | CG iterations per solve (small values suffice with warm-start) |
| `--conn-col-chunk` | 32 | Columns per CG chunk (larger amortizes kernel-launch overhead) |
| `--conn-edge-chunk` | 0 | Bound the CG matvec temporary to `[edge_chunk, col_chunk]` rows — use on large designs to avoid OOM |
| `--early-stop-tol F` | 0 | Stop when overflow improves < F (e.g. `0.01` = 1%) over `--early-stop-patience` outer iters, once rho has reached `rho_max`. Adapts the iteration count to the design instead of a fixed `--max-iterations` (which is inherently design-size-dependent). Free: overflow is already computed for the λ update |
| `--early-stop-patience N` | 3 | Outer iters the early-stop improvement is measured over |
| `--conn-freeze-outer N` | 0 | Stop evaluating connectivity after outer iter N. ⚠️ Not recommended: the later congestion phase then degrades connectivity |
| `--conn-max-sinks K` | 0 | Cap connectivity to the first K sinks per net |
| `--conn-bf16` | off | Reserved (solver is launch-bound, so limited benefit) |

Use `--conn-every 5` for maximum connectivity; add `--conn-super-sink --conn-multi-gpu 4`
for the fastest, least-congested route.

### Objective weights

| Option | Default | Description |
|--------|---------|-------------|
| `--w-wl` | 1.0 | Wirelength weight |
| `--w-conn` | 1.0 | Connectivity (effective resistance) weight |
| `--w-flow` | 1.0 | Flow conservation (directed mode only) |
| `--w-disc` | 0.0 | Discretization `Σx(1-x)`. Found ineffective — the congestion penalty pins `x` below 0.5, so this only shrinks weights (see `--disc-ramp-outer`) |
| `--rho`, `--lr-x`, `--lr-lam` | 1.0 / 0.01 / 0.1 | AL penalty + step sizes |

### Guide export (for Potter)

| Script | Guide content | Use |
|--------|---------------|-----|
| **`scripts/gpu_guide_export.py`** | **Tight per-net shortest path** (~10 tiles/net) via GPU batched Bellman-Ford — all nets relaxed simultaneously | **both modes (recommended)** |
| `scripts/export_potter_guidance.py --guide-out` | Same paths via per-net Dijkstra (CPU, multi-process). Reference implementation — far slower | validation |
| `scripts/fast_guide_export.py` | Whole above-threshold corridor via GPU scatter. Fast but ~9× looser | experimental — see below |

**Guide tightness is the guidance.** A loose corridor guide (~90 tiles/net) behaves like
*no guide at all*: a soft A\* penalty spread over a large tile cloud does not steer the
search, so Potter's runtime and CPWL match the unguided baseline. `fast_guide_export.py`
saves export time and gives it straight back in detailed routing; it is kept only for
experimentation (`--rel` tightens the corridor via a per-net relative threshold).

`gpu_guide_export.py` gets tight paths *and* speed: every net's subgraph is a disjoint
block of the flattened node array, so one `scatter_reduce(amin)` relaxes all nets at
once (Bellman-Ford), then a vectorised backward walk reconstructs every path. Edge cost
is `1/(x+eps)`, so paths follow the global route; nets with negligible flow still get a
sensible path (no separate fallback needed).

`export_potter_guidance.py --out` additionally writes per-connection **bboxes** (for
Potter's `PartitionTree` scheduler), but that path is single-threaded per-net Dijkstra —
slow on large designs; omit unless needed.

Output layout:

```
results/<testcase>/
  checkpoint/global_x.pt
  checkpoint/congestion_evolution.gif  (with --viz)
  tile_paths.json
  <testcase>_routed.phys
  <testcase>_routed.dcp
  metrics.json
```

### Evaluation (`eval_circuits.py`)

```bash
python eval_circuits.py --testcases boom_med_pb boom_soc_v2
```

---

## Module Layout

| Module | Role |
|--------|------|
| `src/rrg/rrg.py` | Unified tile RRG (`RRG`; phys + optional directed view) |
| `src/router/net_index.py` | Pre-built net list / bbox edge cache |
| `src/router/global_router.py` | GlobalRouter losses |
| `src/router/flow_conservation.py` | Per-node Kirchhoff flow penalty |
| `src/router/meng_lambda.py` | Meng normalized subgradient λ update |
| `src/router/augmented_lagrangian.py` | AL optimizer |
| `src/router/connectivity.py` | Effective resistance + batched CG (legacy; `--connectivity-solver cg`) |
| `src/router/connectivity_grouped.py` | **Net-grouped subgraph CG** (`grouped`): super-sink RHS, warm-start, multi-GPU |
| `src/router/route_extractor.py` | Discrete tile paths (parallel across cores) |
| `src/detailed_route.py` | RapidWright PartialRouter |
| `src/io/write_design.py` | Write `.phys`/`.dcp` |
| `cpp/` | C++ Interchange RRG extractor (`extract_rrg`) |
| `scripts/PrebuildNetIndex.py` | Offline net index cache (C++ stub nets + RRG bbox edges) |
| `cpp/build/extract_net_index` | Potter-like stub net + INT tile bbox extraction from `.phys` |
| `scripts/analyze_rrg_distances.py` | Validate RRG Manhattan distances |
| `scripts/gpu_guide_export.py` | **Potter guide: GPU batched Bellman-Ford shortest paths (all nets at once)** |
| `scripts/export_potter_guidance.py` | Potter guide: per-net Dijkstra paths + per-connection bboxes (CPU reference) |
| `scripts/fast_guide_export.py` | Potter guide: per-net corridor via GPU scatter (loose; experimental) |
| `scripts/run_all_benchmarks.sh` | End-to-end campaign over benchmarks, both modes → `campaign/results.csv` |
| `potter/setup_potter.sh` | Clone Potter at the pinned commit, apply the GR-guidance patch, build |
| `potter/0001-diffrouter-gr-guidance.patch` | The Potter-side patch (`-g/--guide`, `--guide_penalty`, soft guide cost in A*) |
| `scripts/eval_connectivity.py` | Fast quality metric: sink-pair connectivity + overflow (GPU label-prop) |
| `docs/PAPER_ALIGNMENT.md` | Paper ↔ code mapping |

## Testing

```bash
python -m pytest tests/ -v
```

Toy-grid tests cover weighted wirelength, per-node flow conservation, Meng λ update, connectivity, AL smoke, and autograd.

Pre-build the net index once per testcase before running `run_exp.py` (see [Default Setup](#default-setup-recommended)). Use `--live-build-nets` only for debugging (loads Java design and builds net list live).

---

## CLI Reference

### Global Router (`src/GlobalRoute.py`)


| Option             | Default                    | Description                                        |
| ------------------ | -------------------------- | -------------------------------------------------- |
| `--testcase`       | boom_soc_v2                | Testcase name                                      |
| `--data`           | ./data/                    | Data directory                                     |
| `--rrg`            | data/rrg_xcvu3p_int.pt | RRG file path (C++ INT tile extract) |
| `--from-device`    | -                          | Build RRG from RapidWright (no pre-extracted file) |
| `--device`         | xcvu3p-ffvc1517-2-e        | Device name                                        |
| `--viz-dir`        | results/checkpoint         | Save congestion GIF                                |
| `--max-iterations` | 1000                       | Max optimization iterations                        |
| `--viz-interval`   | 50                         | Dump congestion frame every N iters                |
| `--cpu`            | -                          | Force CPU                                          |
| `--rrg-log`        | rrg_capacity.log           | RRG capacity log path                              |
| `--rrg-only`       | -                          | Only write RRG capacity log, then exit             |


**Bbox modes:**


| Option        | Description                                                                     |
| ------------- | ------------------------------------------------------------------------------- |
| `--bbox-only` | Output bbox distribution and exit. Loads RRG, plots INT tiles.                  |
| `--bbox-fast` | With `--bbox-only`: skip RRG load (fast, no INT tiles)                          |
| `--bbox-out`  | Save per-net bbox to JSON file                                                  |
| `--bbox-plot` | Save bbox distribution plot (default: results/checkpoint/bbox_distribution.png) |


**Examples:**

```bash
# Default run (pre-extracted RRG)
python src/GlobalRoute.py

# Custom RRG path
python src/GlobalRoute.py --rrg data/rrg_xcvu3p_int.pt

# Bbox distribution (loads RRG, plots tile_col, tile_row, area, INT tiles)
python src/GlobalRoute.py --bbox-only

# Bbox fast (no RRG, plot only tile_col, tile_row, area)
python src/GlobalRoute.py --bbox-only --bbox-fast

# Bbox with JSON + custom plot path
python src/GlobalRoute.py --bbox-only --bbox-out bbox.json --bbox-plot results/bbox_hist.png

# Build RRG from device
python src/GlobalRoute.py --from-device
```

---

### Extract RRG (`scripts/ExtractRRG.py`)

**Recommended: C++ Interchange extractor** (reads `data/xcvu3p.device` directly, no RapidWright).

| Option | Default | Description |
|--------|---------|-------------|
| `--extractor` | cpp | `cpp` (Interchange .device) or `rapidwright` (legacy) |
| `--device-file` | data/xcvu3p.device | FPGA Interchange device file |
| `--cpp-binary` | cpp/build/extract_rrg | Path to C++ extractor |
| `--from-json` | - | Pack existing JSON to output (skip extract) |
| `--output`, `-o` | data/rrg_*.pt | Output path (.pt or .json) |

**Legacy RapidWright options** (only with `--extractor rapidwright`):

| Option | Description |
|--------|-------------|
| `--device` | RapidWright device name |
| `--int-only` | INT tiles only |
| `--adjacent-only` | Adjacent edges only (no jumps) |
| `--no-capacity` | Capacity=1 for all edges |

**Examples:**

```bash
# C++ extract -> .pt (runs cpp/build/extract_rrg, then packs JSON)
python scripts/ExtractRRG.py --extractor cpp --device-file data/xcvu3p.device -o data/rrg_xcvu3p_int.pt

# Pack pre-built JSON
python scripts/ExtractRRG.py --from-json data/rrg_xcvu3p_int_interchange.json -o data/rrg_xcvu3p_int.pt

# Manual C++ + pack
cpp/build/extract_rrg -i data/xcvu3p.device -o data/rrg_xcvu3p_int_interchange.json
python scripts/ExtractRRG.py --from-json data/rrg_xcvu3p_int_interchange.json -o data/rrg_xcvu3p_int.pt

# Analyze distances
python scripts/analyze_rrg_distances.py data/rrg_xcvu3p_int.pt

# Legacy RapidWright adjacent-only
python scripts/ExtractRRG.py --extractor rapidwright --int-only --adjacent-only -o data/rrg_xcvu3p_int_adj.pt
```

### C++ RRG Extractor (`cpp/`)

Build (uses Cap'n Proto from `POTTER_ROOT` if set, e.g. `/path/to/Potter`):

```bash
make -C cpp build
# or: cmake -B cpp/build -DPOTTER_ROOT=/path/to/Potter cpp && cmake --build cpp/build -j
```

The extractor:
- Loads gzip FPGA Interchange `DeviceResources` from `.device`
- Uses **exact `INT` tile type** nodes (not `INT_INTF_*`)
- Aggregates **general** routing nodes into tile edges at H/V distances `{1,2,4,12}`
- Excludes global/clock wire categories
- Writes JSON compatible with `scripts/ExtractRRG.py --from-json`

Requires ~2GB RAM during device parse. First run on `xcvu3p.device` takes ~1 minute.

---

### Visualizer (`src/Visualizer.py`)


| Option       | Default     | Description                      |
| ------------ | ----------- | -------------------------------- |
| `--testcase` | boom_soc_v2 | Testcase name                    |
| `--data`     | ./data/     | Data directory                   |
| `--density`  | -           | Density map instead of net lines |
| `--max-nets` | all         | Max nets to draw                 |


**Examples:**

```bash
python src/Visualizer.py
python src/Visualizer.py --density --testcase boom_med_pb
```

---

## Legacy / debugging

Older paths kept for comparison and debugging. Prefer the [default setup](#default-setup-recommended) above.

**Adjacent-only RRG** (no long-line jumps at distances 2/4/12):

```bash
python scripts/ExtractRRG.py --extractor rapidwright --int-only --adjacent-only \
  -o data/rrg_xcvu3p_int_adj.pt
```

**All signal nets via Java/RapidWright** (not Potter stub filter):

```bash
python scripts/PrebuildNetIndex.py --testcase boom_soc_v2 --use-java
```

**Live net list from Java design** (slow; for debugging only):

```bash
python run_exp.py --testcase boom_soc_v2 --live-build-nets
```

---

## Data Layout

```
data/
  xcvu3p.device              # FPGA Interchange device (input to C++ extractor)
  rrg_xcvu3p_int.pt         # Default RRG (INT tiles, jumps at {1,2,4,12})
  rrg_xcvu3p_int_interchange.json
  rrg_xcvu3p_int_adj.pt     # Legacy: RapidWright adjacent-only RRG
  boom_soc_v2/
    boom_soc_v2_unrouted.phys
    net_index/               # Pre-built stub net caches (see Default Setup)
    ...
```

## Requirements

- Python 3.9+ with PyTorch, matplotlib, Pillow (e.g. `miniforge3/envs/diffrouter`)
- **C++ RRG / net index:** Cap'n Proto + zlib (`make -C cpp build`; builds `extract_rrg` and `extract_net_index`; uses Potter's capnproto if `POTTER_ROOT` is set)
- RapidWright (optional: legacy RRG extract, detailed routing, design load)

```bash
# Unit test (toy 3x3 grid)
python tests/test_toy_al.py
```

