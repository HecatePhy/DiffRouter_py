#!/usr/bin/env python3
"""
Extract or pack RRG (Routing Resource Graph) for DiffRouter.

Sources:
  - C++ Interchange extractor (recommended): reads FPGA Interchange .device directly
  - RapidWright (legacy): uses getWireConnections tile closure

Formats:
  - .pt (PyTorch): native tensors, fast load, best for router
  - .json: human-readable, portable

Usage:
  # C++ extract from Interchange device -> JSON -> .pt
  python scripts/ExtractRRG.py --extractor cpp --device-file data/xcvu3p.device -o data/rrg_xcvu3p_int.pt

  # Pack existing C++ JSON output
  python scripts/ExtractRRG.py --from-json data/rrg_xcvu3p_int_interchange.json -o data/rrg_xcvu3p_int.pt

  # Legacy RapidWright extract
  python scripts/ExtractRRG.py --extractor rapidwright --device xcvu3p-ffvc1517-2-e --int-only --adjacent-only -o data/rrg_xcvu3p_int_adj.pt
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_rrg_json(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def run_cpp_extractor(device_file: str, json_output: str, cpp_binary: str) -> None:
    if not os.path.isfile(cpp_binary):
        raise FileNotFoundError(
            f"C++ extractor not found: {cpp_binary}\n"
            "Build with: make -C cpp build  (set POTTER_ROOT if needed)"
        )
    os.makedirs(os.path.dirname(json_output) or ".", exist_ok=True)
    cmd = [cpp_binary, "-i", device_file, "-o", json_output]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def extract_rrg(
    device_name: str,
    compute_capacity: bool = True,
    verbose_capacity: bool = False,
    int_only: bool = False,
    adjacent_only: bool = False,
) -> dict:
    """Extract RRG from RapidWright Device (legacy path)."""
    from src.load_design import (
        get_device,
        get_tile_graph,
        get_tile_edge_capacities,
        get_int_tile_graph,
        get_int_tile_edge_capacities,
        build_coord_to_int_mapping,
    )

    device = get_device(device_name)
    if int_only:
        print("Extracting INT-only RRG (smaller graph)...")
        int_graph = get_int_tile_graph(device)
        tile_graph = {
            "tiles": [(r, c, t, n, True) for r, c, t, n in int_graph["int_tiles"]],
            "coord_to_idx": int_graph["int_tile_idx"],
            "edges": int_graph["int_edges"],
        }
        if adjacent_only:
            all_edges = sorted(tile_graph["edges"])
            if compute_capacity:
                print("Computing capacities for adjacent INT edges only...")
                cap_full, wire_edges = get_int_tile_edge_capacities(
                    device, int_graph, verbose=verbose_capacity
                )
                edge_capacities = {e: cap_full.get(e, 1) for e in all_edges}
            else:
                edge_capacities = {tuple(e): 1 for e in all_edges}
            print(f"  Adjacent-only: {len(all_edges)} edges (no jump)")
        elif compute_capacity:
            print("Computing INT edge capacities (wire-based)...")
            edge_capacities, wire_edges = get_int_tile_edge_capacities(
                device, int_graph, verbose=verbose_capacity
            )
            all_edges = sorted(set(tile_graph["edges"]) | wire_edges)
            n_jump = len(all_edges) - len(tile_graph["edges"])
            if n_jump > 0:
                print(f"  Wire-based edges: {len(all_edges)} total ({n_jump} jump)")
        else:
            edge_capacities = {tuple(e): 1 for e in tile_graph["edges"]}
            all_edges = tile_graph["edges"]

        print("Building coord_to_int (pin tile -> INT tile mapping)...")
        coord_to_int = build_coord_to_int_mapping(device, int_graph["int_tile_idx"])
        coord_to_int_serial = {f"{r}_{c}": idx for (r, c), idx in coord_to_int.items()}
    else:
        tile_graph = get_tile_graph(device)
        if compute_capacity:
            print("Computing edge capacities (this may take a while)...")
            edge_capacities, wire_edges = get_tile_edge_capacities(
                device, tile_graph, verbose=verbose_capacity
            )
            all_edges = sorted(set(tile_graph["edges"]) | wire_edges)
            n_jump = len(all_edges) - len(tile_graph["edges"])
            if n_jump > 0:
                print(f"  Wire-based edges: {len(all_edges)} total ({n_jump} jump/long-wire)")
        else:
            edge_capacities = {tuple(e): 1 for e in tile_graph["edges"]}
            all_edges = tile_graph["edges"]
        coord_to_int_serial = None

    tiles = []
    for row, col, tile_obj, name, is_int in tile_graph["tiles"]:
        tiles.append([int(row), int(col), str(name), bool(is_int)])

    edges = [[int(a), int(b)] for a, b in all_edges]
    cap_dict = {}
    for (a, b), cap in edge_capacities.items():
        cap_dict[f"{a}_{b}"] = int(cap)
    for a, b in all_edges:
        key = f"{a}_{b}"
        if key not in cap_dict:
            cap_dict[key] = 1

    result = {
        "device_name": device_name,
        "device_rows": int(device.getRows()),
        "device_cols": int(device.getColumns()),
        "tiles": tiles,
        "edges": edges,
        "edge_capacities": cap_dict,
    }
    if coord_to_int_serial is not None:
        result["coord_to_int"] = coord_to_int_serial
        result["int_only"] = True
    return result


def save_rrg_json(rrg_data: dict, output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(rrg_data, f, indent=2)
    n_tiles = len(rrg_data["tiles"])
    n_edges = len(rrg_data["edges"])
    extra = f", coord_to_int: {len(rrg_data.get('coord_to_int', {}))} mappings" if rrg_data.get("int_only") else ""
    print(f"RRG saved to: {output_path} (JSON)")
    print(f"  Tiles: {n_tiles}, Edges: {n_edges}{extra}")


def save_rrg_pt(rrg_data: dict, output_path: str) -> None:
    import torch
    from src.load_design import edge_fabric_geometry

    tiles = rrg_data["tiles"]
    edges = rrg_data["edges"]
    cap_dict = rrg_data["edge_capacities"]
    dist_dict = rrg_data.get("edge_distances", {})
    wl_dict = rrg_data.get("edge_wl_scores", {})

    tile_arr = torch.tensor(
        [[t[0], t[1], int(t[3]), 0] for t in tiles],
        dtype=torch.long,
    )
    edge_arr = torch.tensor(edges, dtype=torch.long)
    cap_arr = torch.tensor(
        [cap_dict.get(f"{a}_{b}", 1) for a, b in edges],
        dtype=torch.float32,
    )
    dist_list = []
    wl_list = []
    tile_tuples = [(int(t[0]), int(t[1]), None, "", bool(t[3])) for t in tiles]
    for a, b in edges:
        key = f"{a}_{b}"
        if key in dist_dict and key in wl_dict:
            dist_list.append(int(dist_dict[key]))
            wl_list.append(int(wl_dict[key]))
        else:
            d, w = edge_fabric_geometry(tile_tuples, int(a), int(b))
            dist_list.append(d)
            wl_list.append(w)
    dist_arr = torch.tensor(dist_list, dtype=torch.long)
    wl_arr = torch.tensor(wl_list, dtype=torch.long)

    pt_data = {
        "format_version": 2,
        "device_name": rrg_data["device_name"],
        "device_rows": rrg_data["device_rows"],
        "device_cols": rrg_data["device_cols"],
        "tiles": tile_arr,
        "edges": edge_arr,
        "edge_capacities": cap_arr,
        "edge_distances": dist_arr,
        "edge_wl_scores": wl_arr,
    }
    if rrg_data.get("int_interchange"):
        pt_data["int_interchange"] = torch.tensor(rrg_data["int_interchange"], dtype=torch.long)
    if rrg_data.get("int_only") and "coord_to_int" in rrg_data:
        pt_data["coord_to_int"] = rrg_data["coord_to_int"]
        pt_data["int_only"] = True

    torch.save(pt_data, output_path)
    extra = f", coord_to_int: {len(rrg_data.get('coord_to_int', {}))}" if rrg_data.get("int_only") else ""
    print(f"RRG saved to: {output_path} (PyTorch)")
    print(f"  Tiles: {tile_arr.shape[0]}, Edges: {edge_arr.shape[0]}{extra}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_cpp = os.path.join(root, "cpp", "build", "extract_rrg")

    parser = argparse.ArgumentParser(description="Extract or pack RRG for DiffRouter")
    parser.add_argument("--extractor", choices=["cpp", "rapidwright"], default="cpp",
                        help="RRG source: cpp (Interchange .device) or rapidwright (legacy)")
    parser.add_argument("--device-file", help="FPGA Interchange .device file (for --extractor cpp)")
    parser.add_argument("--cpp-binary", default=default_cpp, help="Path to cpp/build/extract_rrg")
    parser.add_argument("--from-json", help="Pack existing RRG JSON (skip extraction)")
    parser.add_argument("--device", default="xcvu3p-ffvc1517-2-e", help="RapidWright device name (legacy)")
    parser.add_argument("--output", "-o", help="Output file (default: data/rrg_<device>.pt)")
    parser.add_argument("--format", choices=["pt", "json"], default=None,
                        help="Format (default: from .pt/.json extension)")
    parser.add_argument("--no-capacity", action="store_true",
                        help="RapidWright only: use capacity=1 for all edges")
    parser.add_argument("--int-only", action="store_true", help="RapidWright only: INT tiles")
    parser.add_argument("--adjacent-only", action="store_true", help="RapidWright only: adjacent INT edges")
    parser.add_argument("--verbose", "-v", action="store_true", help="RapidWright only: verbose capacity")
    args = parser.parse_args()

    output = args.output
    if not output:
        if args.device_file:
            base = os.path.splitext(os.path.basename(args.device_file))[0]
            output = f"data/rrg_{base}_int.pt"
        else:
            safe_name = args.device.replace("-", "_").replace(".", "_")
            output = f"data/rrg_{safe_name}.pt"

    fmt = args.format
    if fmt is None:
        fmt = "pt" if output.endswith(".pt") else "json"

    if args.from_json:
        rrg_data = load_rrg_json(args.from_json)
    elif args.extractor == "cpp":
        device_file = args.device_file or os.path.join(root, "data", "xcvu3p.device")
        if fmt == "json" and output.endswith(".json"):
            json_out = output
        else:
            json_out = output.rsplit(".", 1)[0] + "_interchange.json"
        run_cpp_extractor(device_file, json_out, args.cpp_binary)
        rrg_data = load_rrg_json(json_out)
    else:
        if args.no_capacity:
            print("Using --no-capacity: edge capacity = 1 for all edges (fast mode)")
        rrg_data = extract_rrg(
            args.device,
            compute_capacity=not args.no_capacity,
            verbose_capacity=args.verbose,
            int_only=args.int_only,
            adjacent_only=args.adjacent_only,
        )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    if fmt == "pt":
        save_rrg_pt(rrg_data, output)
    else:
        save_rrg_json(rrg_data, output)


if __name__ == "__main__":
    main()
