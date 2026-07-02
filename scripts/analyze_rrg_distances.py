#!/usr/bin/env python3
"""
Analyze Manhattan distances between INT tile nodes in a pre-extracted RRG (.pt or .json).

Tile coordinates in C++-extracted RRGs are fabric lattice indices (Y, X from INT_X#Y#).
Interchange device grid coords are stored separately as int_interchange for pin mapping.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


def load_rrg(path: Path):
    if path.suffix == ".pt":
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        tiles = data["tiles"]
        edges = data["edges"]
        caps = data["edge_capacities"]
        meta = {
            "device_name": data.get("device_name", "?"),
            "device_rows": int(data["device_rows"]),
            "device_cols": int(data["device_cols"]),
            "fabric_coords": "int_interchange" in data,
        }
        return tiles, edges, caps, meta

    with open(path) as f:
        data = json.load(f)
    tiles_list = data["tiles"]
    tiles = torch.tensor([[t[0], t[1], int(t[3]), 0] for t in tiles_list], dtype=torch.long)
    edges = torch.tensor(data["edges"], dtype=torch.long)
    cap_dict = data["edge_capacities"]
    caps = torch.tensor([cap_dict.get(f"{a}_{b}", 1) for a, b in data["edges"]], dtype=torch.float32)
    meta = {
        "device_name": data.get("device_name", "?"),
        "device_rows": int(data["device_rows"]),
        "device_cols": int(data["device_cols"]),
        "fabric_coords": "int_interchange" in data,
    }
    return tiles, edges, caps, meta


def edge_geometry(tiles, a: int, b: int):
    ra, ca = int(tiles[a, 0]), int(tiles[a, 1])
    rb, cb = int(tiles[b, 0]), int(tiles[b, 1])
    dr, dc = abs(ra - rb), abs(ca - cb)
    man = dr + dc
    if dr == 0:
        orient = "H"
    elif dc == 0:
        orient = "V"
    else:
        orient = "diag"
    return man, orient


def classify(man: int, orient: str, allowed: set[int]) -> str:
    if man in allowed and orient in ("H", "V"):
        return f"valid_{man}_{orient}"
    if man in allowed and orient == "diag":
        return "valid_dist_diag"
    if orient in ("H", "V"):
        return f"bad_dist_{orient}"
    return "bad_dist_diag"


def main():
    parser = argparse.ArgumentParser(description="Analyze RRG edge Manhattan distances")
    parser.add_argument("rrg_path", type=Path, help="Path to rrg_*.pt or *.json")
    parser.add_argument("--allowed", type=int, nargs="+", default=[1, 2, 4, 12],
                        help="Architecturally valid Manhattan distances")
    parser.add_argument("--csv", type=Path, help="Write distance histogram CSV")
    parser.add_argument("--examples", type=int, default=3, help="Examples per category")
    args = parser.parse_args()

    allowed = set(args.allowed)
    tiles, edges, caps, meta = load_rrg(args.rrg_path)
    n = tiles.shape[0]
    m = edges.shape[0]

    dist_counter = Counter()
    orient_at_dist = defaultdict(Counter)
    cat_counter = Counter()
    cap_by_cat = defaultdict(list)
    examples = defaultdict(list)

    for k in range(m):
        a, b = int(edges[k, 0]), int(edges[k, 1])
        cap = float(caps[k])
        man, orient = edge_geometry(tiles, a, b)
        dist_counter[man] += 1
        orient_at_dist[man][orient] += 1
        cat = classify(man, orient, allowed)
        cat_counter[cat] += 1
        cap_by_cat[cat].append(cap)
        if len(examples[cat]) < args.examples:
            ra, ca = int(tiles[a, 0]), int(tiles[a, 1])
            rb, cb = int(tiles[b, 0]), int(tiles[b, 1])
            examples[cat].append((a, b, ra, ca, rb, cb, man, orient, cap))

    valid_hv = sum(
        cnt for cat, cnt in cat_counter.items()
        if cat.startswith("valid_") and "diag" not in cat
    )
    unknown = m - valid_hv

    print("=" * 72)
    print(f"RRG: {args.rrg_path}")
    print(f"Device: {meta['device_name']}  grid {meta['device_rows']}x{meta['device_cols']}")
    if meta.get("fabric_coords"):
        print("Tile coords: fabric Y/X (INT_X#Y# lattice)")
    print("=" * 72)
    print(f"INT tiles:           {n:,}")
    print(f"Edges:               {m:,}  (edges/n={m/n:.2f}, mean degree={2*m/n:.2f})")
    print(f"Architectural bound: |E| <= 8n = {8*n:,}")
    print(f"Allowed distances:   {sorted(allowed)}")
    print()
    print("Category summary:")
    for cat, cnt in sorted(cat_counter.items(), key=lambda x: -x[1]):
        c = cap_by_cat[cat]
        print(
            f"  {cat:20s} edges={cnt:7,} ({100*cnt/m:5.1f}%) "
            f"cap mean={statistics.mean(c):6.1f} min={min(c):.0f} max={max(c):.0f}"
        )
    print()
    print(f"Valid H/V @ allowed dist: {valid_hv:,} ({100*valid_hv/m:.1f}%)")
    print(f"Unknown / suspicious:     {unknown:,} ({100*unknown/m:.1f}%)")
    print()

    print("Distance histogram:")
    for dist in sorted(dist_counter):
        o = orient_at_dist[dist]
        mark = " <-- allowed" if dist in allowed else ""
        print(
            f"  dist={dist:3d}: {dist_counter[dist]:7,}  "
            f"H={o['H']:6,} V={o['V']:6,} diag={o['diag']:6,}{mark}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["manhattan_dist", "count", "H", "V", "diag", "allowed"])
            for dist in sorted(dist_counter):
                o = orient_at_dist[dist]
                w.writerow([dist, dist_counter[dist], o["H"], o["V"], o["diag"], dist in allowed])
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
