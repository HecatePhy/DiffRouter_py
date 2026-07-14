"""Pre-built net list and per-net bbox edge mappings for fast GlobalRouter startup."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


def get_net_tiles(net: Any):
    tiles = set()
    for pin in net.getPins():
        tile = pin.getTile()
        if tile is not None:
            tiles.add((int(tile.getRow()), int(tile.getColumn())))
    return tiles


def file_fingerprint(*paths: str) -> str:
    """SHA256 over concatenated file contents (missing paths skipped)."""
    h = hashlib.sha256()
    for path in paths:
        if path and os.path.isfile(path):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def default_device_path(data_prefix: str = "./data/") -> str:
    return os.path.join(data_prefix, "xcvu3p.device")


def find_extract_net_index() -> Optional[str]:
    """Locate extract_net_index binary (cpp/build or PATH)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(root, "cpp", "build", "extract_net_index"),
        os.path.join(root, "cpp", "build", "Release", "extract_net_index"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    from shutil import which

    return which("extract_net_index")


def default_net_index_path(
    data_prefix: str,
    testcase: str,
    rrg_path: str,
    edge_mode: str,
    min_fanout: int,
    expansion_ratio: float,
    route_filter: str = "stubs",
    edge_scope: str = "corridor",
    corridor_width: int = 2,
    max_edges_per_net: Optional[int] = 50000,
) -> str:
    rrg_stem = os.path.splitext(os.path.basename(rrg_path))[0]
    exp_tag = int(round(expansion_ratio * 100))
    filter_tag = route_filter if route_filter else "all"
    name = f"{rrg_stem}_{edge_mode}_{filter_tag}_mf{min_fanout}_exp{exp_tag}"
    if edge_scope == "corridor":
        name += f"_corr{corridor_width}"
        if max_edges_per_net is not None:
            name += f"_cap{max_edges_per_net}"
    else:
        name += "_bbox"
    return os.path.join(data_prefix, testcase, "net_index", f"{name}.pt")


def resolve_net_edges(
    rrg: Any,
    edge_mode: str,
    edge_scope: str,
    src_idx: int,
    sink_idxs: List[int],
    bbox: Tuple[int, int, int, int],
    corridor_width: int = 2,
    max_edges_per_net: Optional[int] = 50000,
) -> Optional[List[int]]:
    """Return edge variable indices for a net, or None if over cap / empty."""
    if edge_scope == "corridor":
        w = corridor_width
        edge_indices: Optional[List[int]] = None
        while w >= 0:
            if edge_mode == "undirected":
                cand = rrg.get_phys_edges_in_corridor(src_idx, sink_idxs, half_width=w)
            else:
                cand = rrg.get_edges_in_corridor(src_idx, sink_idxs, half_width=w)
            if max_edges_per_net is None or len(cand) <= max_edges_per_net:
                edge_indices = cand
                break
            w -= 1
    else:
        min_col, max_col, min_row, max_row = bbox
        if edge_mode == "undirected":
            edge_indices = rrg.get_phys_edges_in_bbox(min_col, max_col, min_row, max_row)
        else:
            edge_indices = rrg.get_edges_in_bbox(min_col, max_col, min_row, max_row)
        if max_edges_per_net is not None and len(edge_indices) > max_edges_per_net:
            return None

    if not edge_indices:
        return None
    return edge_indices


def _net_name(net: Any) -> str:
    name = net.getName()
    return str(name) if name is not None else ""


@dataclass
class NetIndex:
    """Cached per-design net metadata and bbox-scoped edge variable layout."""

    format_version: int = 2
    rrg_fingerprint: str = ""
    design_fingerprint: str = ""
    route_filter: str = "stubs"
    min_fanout: int = 0
    expansion_ratio: float = 0.1
    edge_mode: str = "directed"
    edge_scope: str = "corridor"
    corridor_width: int = 2
    max_edges_per_net: Optional[int] = 50000
    max_nets: Optional[int] = None
    net_names: List[str] = field(default_factory=list)
    net_src_tile: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    net_sink_tiles: List[torch.Tensor] = field(default_factory=list)
    net_fanout: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    net_bbox: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 4, dtype=torch.long))
    net_edge_indices: List[torch.Tensor] = field(default_factory=list)
    var_offset: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    num_nets: int = 0
    num_vars: int = 0

    def matches(
        self,
        rrg_fingerprint: str,
        design_fingerprint: str,
        min_fanout: int,
        expansion_ratio: float,
        edge_mode: str,
        max_nets: Optional[int],
        route_filter: str = "stubs",
    ) -> bool:
        return (
            self.rrg_fingerprint == rrg_fingerprint
            and self.design_fingerprint == design_fingerprint
            and self.route_filter == route_filter
            and self.min_fanout == min_fanout
            and abs(self.expansion_ratio - expansion_ratio) < 1e-9
            and self.edge_mode == edge_mode
            and self.max_nets == max_nets
        )

    def validate_for_load(self, rrg_path: str, edge_mode: str) -> None:
        if self.edge_mode != edge_mode:
            raise ValueError(
                f"net index edge_mode={self.edge_mode!r} != requested {edge_mode!r}"
            )
        expected = file_fingerprint(rrg_path)
        if self.rrg_fingerprint and expected and self.rrg_fingerprint != expected:
            raise ValueError(
                f"net index RRG fingerprint mismatch (index built for different RRG than {rrg_path})"
            )

    @classmethod
    def build(
        cls,
        design: Any,
        rrg: Any,
        device_rows: int,
        device_cols: int,
        coord_to_int: Optional[dict],
        min_fanout: int = 0,
        expansion_ratio: float = 0.1,
        edge_mode: str = "directed",
        max_nets: Optional[int] = None,
        rrg_fingerprint: str = "",
        design_fingerprint: str = "",
        progress_interval: int = 10000,
        verbose: bool = True,
    ) -> "NetIndex":
        pin_to_idx = coord_to_int if coord_to_int is not None else rrg.coord_to_idx
        net_names: List[str] = []
        net_src: List[int] = []
        net_sinks: List[List[int]] = []
        net_fanout: List[int] = []
        net_bbox: List[Tuple[int, int, int, int]] = []
        net_edge_indices: List[List[int]] = []

        scanned = 0
        for net in design.getNets():
            scanned += 1
            if verbose and progress_interval > 0 and scanned % progress_interval == 0:
                print(f"    Net index: scanned {scanned}, kept {len(net_names)} nets...")
            if net.getSource() is None and not net.isStaticNet():
                continue
            sink_pins = list(net.getSinkPins())
            fanout = len(sink_pins)
            if fanout <= min_fanout:
                continue

            tiles = get_net_tiles(net)
            if len(tiles) < 2:
                continue

            rows = [t[0] for t in tiles]
            cols = [t[1] for t in tiles]
            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)

            w = max(1, max_col - min_col)
            h = max(1, max_row - min_row)
            exp_w = max(1, int(w * expansion_ratio))
            exp_h = max(1, int(h * expansion_ratio))
            min_col = max(0, min_col - exp_w)
            max_col = min(device_cols, max_col + exp_w)
            min_row = max(0, min_row - exp_h)
            max_row = min(device_rows, max_row + exp_h)

            if edge_mode == "undirected":
                edge_indices = rrg.get_phys_edges_in_bbox(min_col, max_col, min_row, max_row)
            else:
                edge_indices = rrg.get_edges_in_bbox(min_col, max_col, min_row, max_row)
            if not edge_indices:
                continue

            src_pin = net.getSource()
            src_tile = src_pin.getTile() if src_pin else None
            if src_tile is None:
                continue
            src_idx = pin_to_idx.get((int(src_tile.getRow()), int(src_tile.getColumn())))
            if src_idx is None:
                continue

            sink_idxs = []
            for pin in sink_pins:
                t = pin.getTile()
                if t is None:
                    continue
                idx = pin_to_idx.get((int(t.getRow()), int(t.getColumn())))
                if idx is not None:
                    sink_idxs.append(idx)
            if not sink_idxs:
                continue

            net_names.append(_net_name(net))
            net_src.append(src_idx)
            net_sinks.append(sink_idxs)
            net_fanout.append(fanout)
            net_bbox.append((min_col, max_col, min_row, max_row))
            net_edge_indices.append(edge_indices)

            if max_nets is not None and len(net_names) >= max_nets:
                break

        var_offset = [0]
        for ei in net_edge_indices:
            var_offset.append(var_offset[-1] + len(ei))

        sink_tensors = [torch.tensor(s, dtype=torch.long) for s in net_sinks]
        edge_tensors = [torch.tensor(e, dtype=torch.long) for e in net_edge_indices]

        return cls(
            format_version=1,
            rrg_fingerprint=rrg_fingerprint,
            design_fingerprint=design_fingerprint,
            min_fanout=min_fanout,
            expansion_ratio=expansion_ratio,
            edge_mode=edge_mode,
            max_nets=max_nets,
            net_names=net_names,
            net_src_tile=torch.tensor(net_src, dtype=torch.long),
            net_fanout=torch.tensor(net_fanout, dtype=torch.long),
            net_bbox=torch.tensor(net_bbox, dtype=torch.long),
            net_sink_tiles=sink_tensors,
            net_edge_indices=edge_tensors,
            var_offset=torch.tensor(var_offset, dtype=torch.long),
            num_nets=len(net_names),
            num_vars=var_offset[-1],
        )

    @classmethod
    def build_from_json(
        cls,
        stub_json: Dict[str, Any],
        rrg: Any,
        edge_mode: str = "directed",
        max_nets: Optional[int] = None,
        rrg_fingerprint: str = "",
        design_fingerprint: str = "",
        edge_scope: str = "corridor",
        corridor_width: int = 2,
        max_edges_per_net: Optional[int] = 50000,
        verbose: bool = True,
    ) -> "NetIndex":
        """Build NetIndex from extract_net_index JSON (Potter-like stub nets)."""
        min_fanout = int(stub_json.get("min_fanout", 0))
        expansion_ratio = float(stub_json.get("expansion_ratio", 0.1))
        route_filter = str(stub_json.get("route_filter", "stubs"))

        net_names: List[str] = []
        net_src: List[int] = []
        net_sinks: List[List[int]] = []
        net_fanout: List[int] = []
        net_bbox: List[Tuple[int, int, int, int]] = []
        net_edge_indices: List[List[int]] = []
        skipped_cap = 0

        for entry in stub_json.get("nets", []):
            if max_nets is not None and len(net_names) >= max_nets:
                break

            bbox = tuple(int(x) for x in entry["bbox"])
            src_idx = int(entry["src_int_idx"])
            sink_idxs = [int(x) for x in entry.get("sink_int_idxs", [])]
            if not sink_idxs:
                continue

            edge_indices = resolve_net_edges(
                rrg,
                edge_mode,
                edge_scope,
                src_idx,
                sink_idxs,
                bbox,
                corridor_width=corridor_width,
                max_edges_per_net=max_edges_per_net,
            )
            if edge_indices is None:
                skipped_cap += 1
                continue

            net_names.append(str(entry["name"]))
            net_src.append(src_idx)
            net_sinks.append(sink_idxs)
            net_fanout.append(int(entry.get("fanout", len(sink_idxs))))
            net_bbox.append(bbox)
            net_edge_indices.append(edge_indices)

        var_offset = [0]
        for ei in net_edge_indices:
            var_offset.append(var_offset[-1] + len(ei))

        if verbose:
            scanned = int(stub_json.get("phys_nets_scanned", 0))
            kept = len(net_names)
            extra = ""
            if skipped_cap:
                extra = f", skipped {skipped_cap} (edge cap)"
            print(
                f"    Stub net list: scanned {scanned} phys nets, kept {kept} nets"
                f" (edge_scope={edge_scope}{extra})"
            )

        return cls(
            format_version=2,
            rrg_fingerprint=rrg_fingerprint,
            design_fingerprint=design_fingerprint,
            route_filter=route_filter,
            min_fanout=min_fanout,
            expansion_ratio=expansion_ratio,
            edge_mode=edge_mode,
            edge_scope=edge_scope,
            corridor_width=corridor_width,
            max_edges_per_net=max_edges_per_net,
            max_nets=max_nets,
            net_names=net_names,
            net_src_tile=torch.tensor(net_src, dtype=torch.long),
            net_fanout=torch.tensor(net_fanout, dtype=torch.long),
            net_bbox=torch.tensor(net_bbox, dtype=torch.long),
            net_sink_tiles=[torch.tensor(s, dtype=torch.long) for s in net_sinks],
            net_edge_indices=[torch.tensor(e, dtype=torch.long) for e in net_edge_indices],
            var_offset=torch.tensor(var_offset, dtype=torch.long),
            num_nets=len(net_names),
            num_vars=var_offset[-1],
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "format_version": self.format_version,
                "rrg_fingerprint": self.rrg_fingerprint,
                "design_fingerprint": self.design_fingerprint,
                "route_filter": self.route_filter,
                "min_fanout": self.min_fanout,
                "expansion_ratio": self.expansion_ratio,
                "edge_mode": self.edge_mode,
                "edge_scope": self.edge_scope,
                "corridor_width": self.corridor_width,
                "max_edges_per_net": self.max_edges_per_net,
                "max_nets": self.max_nets,
                "net_names": self.net_names,
                "net_src_tile": self.net_src_tile,
                "net_fanout": self.net_fanout,
                "net_bbox": self.net_bbox,
                "net_sink_tiles": self.net_sink_tiles,
                "net_edge_indices": self.net_edge_indices,
                "var_offset": self.var_offset,
                "num_nets": self.num_nets,
                "num_vars": self.num_vars,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "NetIndex":
        data = torch.load(path, map_location="cpu")
        return cls(
            format_version=int(data.get("format_version", 1)),
            rrg_fingerprint=data.get("rrg_fingerprint", ""),
            design_fingerprint=data.get("design_fingerprint", ""),
            route_filter=str(data.get("route_filter", "all")),
            min_fanout=int(data["min_fanout"]),
            expansion_ratio=float(data["expansion_ratio"]),
            edge_mode=str(data["edge_mode"]),
            edge_scope=str(data.get("edge_scope", "bbox")),
            corridor_width=int(data.get("corridor_width", 2)),
            max_edges_per_net=data.get("max_edges_per_net"),
            max_nets=data.get("max_nets"),
            net_names=list(data["net_names"]),
            net_src_tile=data["net_src_tile"].long(),
            net_fanout=data["net_fanout"].long(),
            net_bbox=data["net_bbox"].long(),
            net_sink_tiles=[t.long() for t in data["net_sink_tiles"]],
            net_edge_indices=[t.long() for t in data["net_edge_indices"]],
            var_offset=data["var_offset"].long(),
            num_nets=int(data["num_nets"]),
            num_vars=int(data["num_vars"]),
        )

    def attach_nets(self, design: Any) -> List[Any]:
        """Map cached net_names to live RapidWright net objects (one design scan)."""
        name_to_net: Dict[str, Any] = {}
        for net in design.getNets():
            name_to_net[_net_name(net)] = net
        nets: List[Any] = []
        missing = 0
        for name in self.net_names:
            net = name_to_net.get(name)
            if net is None:
                missing += 1
                nets.append(None)
            else:
                nets.append(net)
        if missing:
            print(f"  Warning: {missing} net(s) from index not found in design")
        return nets

    def apply_to_router(self, router: Any, design: Optional[Any] = None) -> None:
        """Populate GlobalRouter fields from this index."""
        dev = router.device
        router.net_names = list(self.net_names)
        router.net_src_tile = self.net_src_tile.tolist()
        router.net_sink_tiles = [t.tolist() for t in self.net_sink_tiles]
        router.net_fanout = self.net_fanout.tolist()
        router.net_bbox = [tuple(row.tolist()) for row in self.net_bbox]
        # Keep as CPU long tensors (list-of-int conversion costs ~30B/var).
        router.net_edge_indices = list(self.net_edge_indices)
        router.num_nets = self.num_nets
        router._var_offset = self.var_offset.tolist()
        router.num_vars = self.num_vars
        router._net_edge_tensors = [
            t.to(dev, dtype=torch.long) for t in self.net_edge_indices
        ]
        if design is not None:
            router.nets = self.attach_nets(design)
        else:
            router.nets = [None] * self.num_nets


def run_extract_net_index(
    device_path: str,
    physical_path: str,
    *,
    expansion_ratio: float = 0.1,
    min_fanout: int = 0,
    max_nets: Optional[int] = None,
    output_json: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Run C++ extract_net_index; return path to stub net list JSON."""
    binary = find_extract_net_index()
    if binary is None:
        raise FileNotFoundError(
            "extract_net_index not found. Build with: make -C cpp build"
        )

    if output_json is None:
        fd, output_json = tempfile.mkstemp(suffix="_stub_nets.json")
        os.close(fd)

    cmd = [
        binary,
        "-d",
        device_path,
        "-i",
        physical_path,
        "-o",
        output_json,
        "--expansion-ratio",
        str(expansion_ratio),
        "--min-fanout",
        str(min_fanout),
    ]
    if max_nets is not None:
        cmd.extend(["--max-nets", str(max_nets)])

    if verbose:
        print(f"    Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_json


def build_and_save(
    rrg_path: str,
    out_path: str,
    physical_path: str,
    *,
    netlist_path: str = "",
    device_path: Optional[str] = None,
    min_fanout: int = 0,
    expansion_ratio: float = 0.1,
    edge_mode: str = "directed",
    max_nets: Optional[int] = None,
    route_filter: str = "stubs",
    edge_scope: str = "corridor",
    corridor_width: int = 2,
    max_edges_per_net: Optional[int] = 50000,
    use_java: bool = False,
    design: Any = None,
    device=None,
    verbose: bool = True,
) -> NetIndex:
    """Offline prebuild: C++ stub extraction (default) or Java/RapidWright fallback."""
    from src.load_design import load_rrg_fast

    if verbose:
        print(f"    Loading RRG from: {rrg_path}")
    rrg, device_rows, device_cols, coord_to_int, fmt = load_rrg_fast(
        rrg_path, edge_mode=edge_mode, device=device
    )
    if verbose and fmt < 2:
        print(
            "    Warning: RRG format_version < 2 (slow load). "
            "Re-pack with: python scripts/ExtractRRG.py --from-json ... -o ..."
        )
    rrg_fingerprint = file_fingerprint(rrg_path)
    design_fingerprint = file_fingerprint(physical_path, netlist_path)

    if use_java or route_filter != "stubs":
        if design is None:
            from src.load_design import load_design

            if verbose:
                print("    Loading design via RapidWright (Java)...")
            design = load_design(physical_path, netlist_path)
        if verbose:
            print(
                f"    Building net index from design (min_fanout={min_fanout}, "
                f"edge_mode={edge_mode}, route_filter={route_filter})..."
            )
        index = NetIndex.build(
            design,
            rrg,
            device_rows,
            device_cols,
            coord_to_int,
            min_fanout=min_fanout,
            expansion_ratio=expansion_ratio,
            edge_mode=edge_mode,
            max_nets=max_nets,
            rrg_fingerprint=rrg_fingerprint,
            design_fingerprint=design_fingerprint,
            verbose=verbose,
        )
        index.route_filter = route_filter
        index.format_version = 1 if route_filter == "all" else 2
    else:
        dev_path = device_path
        if dev_path is None:
            physical_dir = os.path.dirname(os.path.abspath(physical_path))
            data_dir = os.path.dirname(physical_dir)
            dev_path = os.path.join(data_dir, "xcvu3p.device")
        if not os.path.isfile(dev_path):
            dev_path = default_device_path()
        if not os.path.isfile(dev_path):
            raise FileNotFoundError(
                f"Device file not found: {dev_path}\n"
                "Pass --device or place xcvu3p.device under data/"
            )

        json_path = run_extract_net_index(
            dev_path,
            physical_path,
            expansion_ratio=expansion_ratio,
            min_fanout=min_fanout,
            max_nets=max_nets,
            verbose=verbose,
        )
        try:
            with open(json_path) as f:
                stub_json = json.load(f)
            if verbose:
                print(
                    f"    Building net index from stub list "
                    f"(edge_mode={edge_mode})..."
                )
            index = NetIndex.build_from_json(
                stub_json,
                rrg,
                edge_mode=edge_mode,
                max_nets=max_nets,
                rrg_fingerprint=rrg_fingerprint,
                design_fingerprint=design_fingerprint,
                edge_scope=edge_scope,
                corridor_width=corridor_width,
                max_edges_per_net=max_edges_per_net,
                verbose=verbose,
            )
        finally:
            if json_path.startswith(tempfile.gettempdir()):
                try:
                    os.remove(json_path)
                except OSError:
                    pass

    index.save(out_path)
    if verbose:
        print(f"    Saved net index: {out_path} ({index.num_nets} nets, {index.num_vars} vars)")
    return index


def require_net_index_path(
    net_index_path: str,
    prebuild_hint: str,
) -> str:
    if not os.path.isfile(net_index_path):
        raise FileNotFoundError(
            f"Net index not found: {net_index_path}\n"
            f"Pre-build with:\n  {prebuild_hint}"
        )
    return net_index_path
