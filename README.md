# DiffRouter-py

Differentiable global router for FPGA routing (FPGA24 Routing Contest). Uses PyTorch for optimization with Augmented Lagrangian.

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
| `--conn-net-batch` | 0 | Nets per iter for connectivity (0=all) |
| `--w-flow` | 1.0 | Flow conservation soft penalty weight |
| `--connectivity-solver` | solve | `solve` or `cg` |
| `--global-only` | - | Skip detailed routing |
| `--skip-global` | - | Load cached `global_x.pt` |
| `--route-threshold` | 0.01 | Edge threshold for path extraction |
| `--max-nets` | - | Limit nets (quick tests) |
| `--edge-mode` | directed | `directed` or `undirected` (undirected: ~½ vars, no flow term) |
| `--net-index` | auto path | Pre-built net index `.pt` (required unless `--live-build-nets`) |
| `--live-build-nets` | - | Debug: build net list from Java design |
| `--resume` | - | Checkpoint path for global opt |

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
| `src/router/connectivity.py` | Effective resistance + CG |
| `src/router/route_extractor.py` | Discrete tile paths |
| `src/detailed_route.py` | RapidWright PartialRouter |
| `src/io/write_design.py` | Write `.phys`/`.dcp` |
| `cpp/` | C++ Interchange RRG extractor (`extract_rrg`) |
| `scripts/PrebuildNetIndex.py` | Offline net index cache (C++ stub nets + RRG bbox edges) |
| `cpp/build/extract_net_index` | Potter-like stub net + INT tile bbox extraction from `.phys` |
| `scripts/analyze_rrg_distances.py` | Validate RRG Manhattan distances |
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

