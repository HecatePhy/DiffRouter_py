"""
Load FPGA design and get routing resources using RapidWright Python API.

Requires: pip install rapidwright (and Java 8+)

References:
- https://www.rapidwright.io/docs/index.html
- https://www.rapidwright.io/docs/Install_RapidWright_as_a_Python_PIP_Package.html
- https://xilinx.github.io/fpga24_routing_contest/
"""

from typing import Optional, List, Iterator, Tuple, Dict, Any, Set
from collections import Counter
import json


def contest_wl_score(dist: int, horizontal: bool) -> int:
    """FPGA24 contest PIP wirelength score for a fabric hop (wa.py / xcvup_device_data.py)."""
    if dist == 1:
        return 1
    if dist == 2:
        return 5 if horizontal else 3
    if dist == 4:
        return 10 if horizontal else 5
    if dist == 12:
        return 14 if horizontal else 12
    return 0


def edge_fabric_geometry(tiles, a: int, b: int) -> Tuple[int, int]:
    """Return (fabric Manhattan dist, contest wl score) for tiles a, b."""
    ya, xa = int(tiles[a][0]), int(tiles[a][1])
    yb, xb = int(tiles[b][0]), int(tiles[b][1])
    dy, dx = abs(ya - yb), abs(xa - xb)
    dist = dy + dx
    horizontal = dy == 0 and dx > 0
    return dist, contest_wl_score(dist, horizontal)


def _parse_edge_feature_dict(raw: Optional[dict]) -> Dict[Tuple[int, int], int]:
    if not raw:
        return {}
    out: Dict[Tuple[int, int], int] = {}
    for key, val in raw.items():
        a, b = map(int, str(key).split("_"))
        out[(min(a, b), max(a, b))] = int(val)
    return out


def _attach_edge_features(
    tile_graph: dict,
    edge_distances: Optional[Dict[Tuple[int, int], int]] = None,
    edge_wl_scores: Optional[Dict[Tuple[int, int], int]] = None,
) -> None:
    """Populate tile_graph edge_dist and edge_wl_score; compute from tiles if missing."""
    tiles = tile_graph["tiles"]
    edge_dist: Dict[Tuple[int, int], int] = {}
    edge_wl: Dict[Tuple[int, int], int] = {}
    for a, b in tile_graph["edges"]:
        phys = (min(int(a), int(b)), max(int(a), int(b)))
        if edge_distances is not None and phys in edge_distances:
            dist = edge_distances[phys]
        else:
            dist, _ = edge_fabric_geometry(tiles, phys[0], phys[1])
        if edge_wl_scores is not None and phys in edge_wl_scores:
            wl = edge_wl_scores[phys]
        else:
            _, wl = edge_fabric_geometry(tiles, phys[0], phys[1])
        edge_dist[phys] = dist
        edge_wl[phys] = wl
    tile_graph["edge_dist"] = edge_dist
    tile_graph["edge_wl_score"] = edge_wl


def load_design(
    phys_path: str,
    logical_path: Optional[str] = None,
) -> Any:
    """
    Load a design from FPGA Interchange Format files using RapidWright.

    Args:
        phys_path: Path to Physical Netlist (.phys)
        logical_path: Path to Logical Netlist (.netlist). Optional; if omitted,
            the design will have placement/routing but no logical netlist.

    Returns:
        RapidWright Design object with:
        - design.getNets() - all nets
        - design.getCells() - all cells
        - net.getSource(), net.getSinkPins(), net.getPIPs()
    """
    import rapidwright
    from com.xilinx.rapidwright.interchange import PhysNetlistReader, LogNetlistReader

    PhysNetlistReader.CHECK_CONSTANT_ROUTING_AND_NET_NAMING = False
    PhysNetlistReader.CHECK_AND_CREATE_LOGICAL_CELL_IF_NOT_PRESENT = False
    PhysNetlistReader.VALIDATE_MACROS_PLACED_FULLY = False
    PhysNetlistReader.CHECK_MACROS_CONSISTENT = False

    design = PhysNetlistReader.readPhysNetlist(phys_path)
    if logical_path:
        log_netlist = LogNetlistReader.readLogNetlist(logical_path)
        design.setNetlist(log_netlist)
    return design


def load_rrg(rrg_path: str) -> Tuple[dict, Dict[Tuple[int, int], int], int, int]:
    """
    Load RRG from file (produced by scripts/ExtractRRG.py).

    Supports:
        - .pt: PyTorch format (native tensors, fast)
        - .json: JSON format (portable)
        - INT-only RRG: tile_graph includes coord_to_int for pin mapping

    Returns:
        (tile_graph, edge_capacities, device_rows, device_cols)
        - tile_graph: dict with tiles, coord_to_idx, edges, edge_dist, edge_wl_score;
          optionally coord_to_int for INT-only
        - edge_capacities: (idx_a, idx_b) -> capacity for idx_a < idx_b
        - device_rows, device_cols: device dimensions
    """
    if rrg_path.endswith(".pt"):
        return _load_rrg_pt(rrg_path)
    return _load_rrg_json(rrg_path)


def _coord_to_int_from_rrg_data(data: dict) -> Optional[Dict[Tuple[int, int], int]]:
    raw = data.get("coord_to_int")
    if not raw:
        return None
    out: Dict[Tuple[int, int], int] = {}
    for k, v in raw.items():
        r, c = map(int, str(k).split("_"))
        out[(r, c)] = int(v)
    return out


def load_rrg_fast(
    rrg_path: str,
    edge_mode: str = "directed",
    device=None,
):
    """
    Load RRG directly (v2 .pt uses tensor-native path; v1 falls back to tile_graph build).

    Returns:
        (rrg, device_rows, device_cols, coord_to_int, format_version)
    """
    from src.rrg.rrg import RRG

    dev = device or __import__("torch").device("cpu")
    if rrg_path.endswith(".pt"):
        import torch

        try:
            data = torch.load(rrg_path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(rrg_path, map_location="cpu")
        fmt = int(data.get("format_version", 1))
        coord_to_int = _coord_to_int_from_rrg_data(data)
        if fmt >= 2:
            rrg = RRG.from_tensors(data, device=dev, edge_mode=edge_mode)
            return rrg, int(data["device_rows"]), int(data["device_cols"]), coord_to_int, fmt
        tile_graph, edge_capacities, device_rows, device_cols = _load_rrg_pt(rrg_path)
        rrg = RRG(tile_graph, edge_capacities, dev, edge_mode=edge_mode)
        if coord_to_int is None:
            coord_to_int = tile_graph.get("coord_to_int")
        return rrg, device_rows, device_cols, coord_to_int, fmt
    tile_graph, edge_capacities, device_rows, device_cols = _load_rrg_json(rrg_path)
    rrg = RRG(tile_graph, edge_capacities, dev, edge_mode=edge_mode)
    return rrg, device_rows, device_cols, tile_graph.get("coord_to_int"), 1


def _build_tile_graph_from_rrg_data(data: dict) -> dict:
    """Build tile_graph dict from RRG JSON/pt payload."""
    tiles = []
    coord_to_idx = {}
    interchange = data.get("int_interchange")
    for idx, tile_entry in enumerate(data["tiles"]):
        row, col, name, is_int = tile_entry[0], tile_entry[1], tile_entry[2], tile_entry[3]
        tiles.append((int(row), int(col), None, str(name), bool(is_int)))
        if interchange is not None:
            ic_row, ic_col = interchange[idx]
            coord_to_idx[(int(ic_row), int(ic_col))] = idx
        else:
            coord_to_idx[(int(row), int(col))] = idx

    edges = [tuple(e) for e in data["edges"]]
    tile_graph = {"tiles": tiles, "coord_to_idx": coord_to_idx, "edges": edges}
    if data.get("int_only") and "coord_to_int" in data:
        coord_to_int = {}
        for k, v in data["coord_to_int"].items():
            r, c = map(int, str(k).split("_"))
            coord_to_int[(r, c)] = int(v)
        tile_graph["coord_to_int"] = coord_to_int
    return tile_graph


def _load_rrg_json(rrg_path: str) -> Tuple[dict, Dict[Tuple[int, int], int], int, int]:
    """Load RRG from JSON file."""
    with open(rrg_path) as f:
        data = json.load(f)

    tile_graph = _build_tile_graph_from_rrg_data(data)

    edge_capacities = {}
    for key, cap in data["edge_capacities"].items():
        a, b = map(int, key.split("_"))
        edge_capacities[(a, b)] = int(cap)

    _attach_edge_features(
        tile_graph,
        _parse_edge_feature_dict(data.get("edge_distances")),
        _parse_edge_feature_dict(data.get("edge_wl_scores")),
    )

    return tile_graph, edge_capacities, data["device_rows"], data["device_cols"]


def _load_rrg_pt(rrg_path: str) -> Tuple[dict, Dict[Tuple[int, int], int], int, int]:
    """Load RRG from PyTorch .pt file."""
    import torch

    try:
        data = torch.load(rrg_path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(rrg_path, map_location="cpu")
    tile_arr = data["tiles"]
    edge_arr = data["edges"]
    cap_arr = data["edge_capacities"]

    payload = {
        "tiles": [
            [
                int(tile_arr[idx, 0].item()),
                int(tile_arr[idx, 1].item()),
                "",
                bool(tile_arr[idx, 2].item()),
            ]
            for idx in range(tile_arr.shape[0])
        ],
        "edges": [tuple(e.tolist()) for e in edge_arr],
        "int_only": data.get("int_only"),
        "coord_to_int": data.get("coord_to_int"),
    }
    if "int_interchange" in data:
        ic = data["int_interchange"]
        payload["int_interchange"] = [
            [int(ic[idx, 0].item()), int(ic[idx, 1].item())] for idx in range(ic.shape[0])
        ]
    if "edge_distances" in data:
        payload["edge_distances"] = {
            f"{int(a)}_{int(b)}": int(v)
            for (a, b), v in zip(
                [tuple(e.tolist()) for e in edge_arr],
                data["edge_distances"].tolist(),
            )
        }
    if "edge_wl_scores" in data:
        payload["edge_wl_scores"] = {
            f"{int(a)}_{int(b)}": int(v)
            for (a, b), v in zip(
                [tuple(e.tolist()) for e in edge_arr],
                data["edge_wl_scores"].tolist(),
            )
        }

    tile_graph = _build_tile_graph_from_rrg_data(payload)

    edges = tile_graph["edges"]
    edge_capacities = {}
    for (a, b), cap in zip(edges, cap_arr.tolist()):
        edge_capacities[(a, b)] = int(cap)

    _attach_edge_features(
        tile_graph,
        _parse_edge_feature_dict(payload.get("edge_distances")),
        _parse_edge_feature_dict(payload.get("edge_wl_scores")),
    )

    return tile_graph, edge_capacities, int(data["device_rows"]), int(data["device_cols"])


def get_device(device_name: str) -> Any:
    """
    Get a RapidWright Device for routing resources.

    Args:
        device_name: e.g. "xcvu3p-ffvc1517-2-e", "xcvu9p-flgb2104-2-i"

    Returns:
        RapidWright Device with:
        - device.getTiles() - 2D array of tiles
        - device.getTile(row, col)
        - tile.getWireCount(), tile.getWireName(idx)
        - device.getWire(wire_name), wire.getNode()
        - node.getAllDownhillNodes(), node.getAllUphillNodes()
    """
    import rapidwright
    from com.xilinx.rapidwright.device import Device

    return Device.getDevice(device_name)


def get_routing_resources(device: Any) -> dict:
    """
    Extract routing resource graph from a RapidWright Device.

    Returns a dict with:
        - device_rows, device_cols
        - tiles: list of (row, col, tile) for each tile
        - iter_nodes: callable returning (node, tile, wire_name) iterator
        - iter_edges: callable returning (node, downhill_node) iterator
    """
    from com.xilinx.rapidwright.device import Node

    tiles_flat = []
    # getAllTiles() returns an iterable of tiles (HashMap.Values), not a 2D array
    for tile in device.getAllTiles():
        if tile is not None:
            row, col = tile.getRow(), tile.getColumn()
            tiles_flat.append((row, col, tile))

    rows, cols = device.getRows(), device.getColumns()

    def iter_nodes():
        for row, col, tile in tiles_flat:
            for wire_idx in range(tile.getWireCount()):
                wire_name = tile.getWireName(wire_idx)
                node = Node.getNode(tile, wire_idx)
                if node is not None:
                    yield (node, tile, wire_name)

    def iter_edges():
        for node, tile, wire_name in iter_nodes():
            downhill = node.getAllDownhillNodes()
            if downhill is not None:
                for d in downhill:
                    if d is not None:
                        yield (node, d)

    return {
        "device_rows": rows,
        "device_cols": cols,
        "tiles": tiles_flat,
        "iter_nodes": iter_nodes,
        "iter_edges": iter_edges,
    }


def get_net_fanout_distribution(design: Any) -> Tuple[Counter, int, int]:
    """
    Compute fanout (sink pin count) distribution for signal nets.

    Returns:
        (fanout_counter, total_nets, nets_with_source)
        - fanout_counter: Counter mapping fanout -> number of nets
        - total_nets: total net count
        - nets_with_source: nets that have a source (or are static)
    """
    total = 0
    with_source = 0
    fanout_counts: Counter = Counter()
    for net in design.getNets():
        total += 1
        if net.getSource() is None and not net.isStaticNet():
            continue
        with_source += 1
        sink_count = len(list(net.getSinkPins()))
        fanout_counts[sink_count] += 1
    return fanout_counts, total, with_source


def print_net_fanout_stats(design: Any) -> None:
    """Print net fanout distribution statistics."""
    fanout_counts, total, with_source = get_net_fanout_distribution(design)
    print("\n=== Net Fanout Distribution ===")
    print(f"Total nets: {total}")
    print(f"Nets with source (or static): {with_source}")

    # Summary stats
    fanouts = []
    for fanout, count in fanout_counts.items():
        fanouts.extend([fanout] * count)
    if fanouts:
        fanouts_sorted = sorted(fanouts)
        print(f"Min fanout: {min(fanouts_sorted)}, Max fanout: {max(fanouts_sorted)}")
        print(f"Mean fanout: {sum(fanouts) / len(fanouts):.2f}")
        median_idx = len(fanouts_sorted) // 2
        median = fanouts_sorted[median_idx] if len(fanouts_sorted) % 2 else (fanouts_sorted[median_idx - 1] + fanouts_sorted[median_idx]) / 2
        print(f"Median fanout: {median:.1f}")

    # Fanout thresholds
    thresholds = [1, 2, 5, 10, 20, 50, 100, 500, 1000]
    print("\nNets by fanout threshold:")
    for thresh in thresholds:
        count = sum(c for f, c in fanout_counts.items() if f > thresh)
        print(f"  fanout > {thresh:4d}: {count:6d} nets")

    # Histogram (first 20 bins)
    print("\nFanout histogram (first 20 bins):")
    max_count = max(fanout_counts.values()) if fanout_counts else 0
    for fanout in sorted(fanout_counts.keys())[:20]:
        count = fanout_counts[fanout]
        bar_len = int(50 * count / max_count) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  fanout={fanout:4d}: {count:6d} {bar}")


def get_nets_to_route(design: Any) -> List[Tuple[Any, List[Any]]]:
    """
    Get signal nets that need routing (have unrouted sink pins).

    Returns:
        List of (net, sink_pins) for each net with stubs to route.
    """
    result = []
    for net in design.getNets():
        if net.getSource() is None and not net.isStaticNet():
            continue
        if not net.hasPIPs():
            sink_pins = list(net.getSinkPins())
            if sink_pins:
                result.append((net, sink_pins))
    return result


def get_tile_graph(device: Any) -> dict:
    """
    Build tile-level graph with ALL tiles.
    Edges: (1) INT tile <-> other tile; (2) INT tile <-> INT tile.
    No edges between two non-INT tiles.

    Returns a dict with:
        - tiles: list of (row, col, tile, name, is_int) for each tile
        - coord_to_idx: (row, col) -> index
        - is_int: list of bool (is tile INT?)
        - edges: list of (idx_a, idx_b) for adjacent tiles (at least one INT)
    """
    all_tiles_list = list(device.getAllTiles())
    tiles = []
    coord_to_idx = {}

    for tile in all_tiles_list:
        if tile is None:
            continue
        name = str(tile.getName())
        row, col = int(tile.getRow()), int(tile.getColumn())
        is_int = name.startswith("INT")
        idx = len(tiles)
        tiles.append((row, col, tile, name, is_int))
        coord_to_idx[(row, col)] = idx

    edges = []
    seen_edges = set()
    for idx, (row, col, tile, name, is_int) in enumerate(tiles):
        for dcol, drow in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor = tile.getTileNeighbor(dcol, drow)
            if neighbor is None:
                continue
            nrow, ncol = int(neighbor.getRow()), int(neighbor.getColumn())
            nidx = coord_to_idx.get((nrow, ncol))
            if nidx is None:
                continue
            neighbor_is_int = str(neighbor.getName()).startswith("INT")
            if not (is_int or neighbor_is_int):
                continue
            edge = (min(idx, nidx), max(idx, nidx))
            if edge not in seen_edges:
                seen_edges.add(edge)
                edges.append((idx, nidx))

    return {
        "tiles": tiles,
        "coord_to_idx": coord_to_idx,
        "is_int": [t[4] for t in tiles],
        "edges": edges,
    }


def get_int_tile_graph(device: Any) -> dict:
    """
    Build a tile-level routing graph abstraction: INT tiles and their connectivity.

    INT (interconnect) tiles contain the programmable switches (PIPs) that route
    signals. This abstraction is much smaller than the full wire-level graph.

    Returns a dict with:
        - total_tiles: total tile count
        - int_tiles: list of (row, col, tile, name) for each INT tile
        - int_tile_count: number of INT tiles
        - int_tile_idx: dict mapping (row, col) -> index in int_tiles
        - int_edges: list of (idx_a, idx_b) for adjacent INT tiles (share a wire)
        - int_adjacency: dict mapping int_tile_idx -> list of neighbor indices
    """
    all_tiles_list = list(device.getAllTiles())
    total_tiles = len(all_tiles_list)

    # Collect INT tiles (name starts with "INT" - includes INT_X, INT_INTF_L, etc.)
    int_tiles = []
    coord_to_idx = {}
    for tile in all_tiles_list:
        if tile is None:
            continue
        name = str(tile.getName())  # Java String -> Python str
        if not name.startswith("INT"):
            continue
        row, col = int(tile.getRow()), int(tile.getColumn())  # Java int -> Python int
        idx = len(int_tiles)
        int_tiles.append((row, col, tile, name))
        coord_to_idx[(row, col)] = idx

    # Build INT tile connectivity: two INT tiles are connected if they are
    # adjacent in the grid and at least one has a neighbor in that direction.
    # Use getTileNeighbor to find adjacent tiles.
    int_edges = []
    seen_edges = set()
    for idx, (row, col, tile, name) in enumerate(int_tiles):
        # Check 4 neighbors: (0,-1), (0,+1), (-1,0), (+1,0)
        for dcol, drow in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor = tile.getTileNeighbor(dcol, drow)
            if neighbor is None:
                continue
            nrow, ncol = int(neighbor.getRow()), int(neighbor.getColumn())
            if not str(neighbor.getName()).startswith("INT"):
                continue
            nidx = coord_to_idx.get((nrow, ncol))
            if nidx is None:
                continue
            edge = (min(idx, nidx), max(idx, nidx))
            if edge not in seen_edges:
                seen_edges.add(edge)
                int_edges.append((idx, nidx))

    # Build adjacency list
    int_adjacency = {i: [] for i in range(len(int_tiles))}
    for a, b in int_edges:
        if b not in int_adjacency[a]:
            int_adjacency[a].append(b)
        if a not in int_adjacency[b]:
            int_adjacency[b].append(a)

    return {
        "total_tiles": total_tiles,
        "int_tiles": int_tiles,
        "int_tile_count": len(int_tiles),
        "int_tile_idx": coord_to_idx,
        "int_edges": int_edges,
        "int_adjacency": int_adjacency,
    }


def get_tile_edge_capacities(
    device: Any,
    tile_graph: dict,
    verbose: bool = False,
) -> Tuple[Dict[Tuple[int, int], int], Set[Tuple[int, int]]]:
    """
    Count PIPs (routing resources) between tiles via getWireConnections (RapidWright API).
    Discovers edges from wire connections, including long-wire/jump connections
    (e.g. 2, 4 tiles apart), not just adjacent tiles from getTileNeighbor.

    Returns:
        (capacity_dict, wire_edges)
        - capacity_dict: (idx_a, idx_b) -> capacity for idx_a < idx_b
        - wire_edges: set of all (idx_a, idx_b) with wire connections
    """
    coord_to_idx = tile_graph["coord_to_idx"]
    tiles_list = tile_graph["tiles"]
    capacity: Dict[Tuple[int, int], int] = {}

    for idx, (row, col, tile, name, is_int) in enumerate(tiles_list):
        for wire_idx in range(tile.getWireCount()):
            conns = tile.getWireConnections(wire_idx)
            if conns is None:
                continue
            for conn_wire in conns:
                if conn_wire is None:
                    continue
                conn_tile = conn_wire.getTile()
                if conn_tile is None or conn_tile == tile:
                    continue
                crow, ccol = int(conn_tile.getRow()), int(conn_tile.getColumn())
                nidx = coord_to_idx.get((crow, ccol))
                if nidx is None:
                    continue
                if idx >= nidx:
                    continue
                edge = (idx, nidx)
                capacity[edge] = capacity.get(edge, 0) + 1
                if verbose:
                    name_a = tiles_list[idx][3]
                    name_b = tiles_list[nidx][3]
                    print(f"  edge {edge} cap={capacity[edge]} | {name_a} ({idx}) <-> {name_b} ({nidx})")

    if verbose:
        n_edges = len(capacity)
        caps = list(capacity.values())
        n_cap1 = sum(1 for c in caps if c == 1)
        print(f"  Total edges with capacity: {n_edges}")
        if caps:
            print(f"  Capacity: min={min(caps)}, max={max(caps)}, mean={sum(caps)/len(caps):.1f}")
            print(f"  Edges with capacity=1: {n_cap1} ({100*n_cap1/n_edges:.1f}%)")

    wire_edges = set(capacity.keys())
    return capacity, wire_edges


def get_int_tile_edge_capacities(
    device: Any, int_graph: dict, verbose: bool = False
) -> Tuple[Dict[Tuple[int, int], int], Set[Tuple[int, int]]]:
    """Get edge capacities for INT-only graph (includes wire-based jump connections)."""
    tile_graph = {
        "tiles": [(r, c, t, n, True) for r, c, t, n in int_graph["int_tiles"]],
        "coord_to_idx": int_graph["int_tile_idx"],
    }
    return get_tile_edge_capacities(device, tile_graph, verbose=verbose)


def build_coord_to_int_mapping(device: Any, int_tile_idx: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    """
    Map every device tile (row,col) to its closest INT tile index.

    For INT tiles: direct mapping. For others: find adjacent INT via getTileNeighbor.
    If no adjacent INT, try expanding search (BFS up to 2 hops).
    """
    coord_to_int: Dict[Tuple[int, int], int] = dict(int_tile_idx)  # INT tiles map to self

    for tile in device.getAllTiles():
        if tile is None:
            continue
        row, col = int(tile.getRow()), int(tile.getColumn())
        if (row, col) in coord_to_int:
            continue
        # Find adjacent INT
        found = None
        for dcol, drow in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor = tile.getTileNeighbor(dcol, drow)
            if neighbor is None:
                continue
            nrow, ncol = int(neighbor.getRow()), int(neighbor.getColumn())
            if str(neighbor.getName()).startswith("INT"):
                idx = int_tile_idx.get((nrow, ncol))
                if idx is not None:
                    found = idx
                    break
        if found is not None:
            coord_to_int[(row, col)] = found
        else:
            # Try 2-hop: neighbor of neighbor
            for dcol, drow in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                neighbor = tile.getTileNeighbor(dcol, drow)
                if neighbor is None:
                    continue
                for dcol2, drow2 in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nn = neighbor.getTileNeighbor(dcol2, drow2)
                    if nn is None:
                        continue
                    if str(nn.getName()).startswith("INT"):
                        nrow, ncol = int(nn.getRow()), int(nn.getColumn())
                        idx = int_tile_idx.get((nrow, ncol))
                        if idx is not None:
                            coord_to_int[(row, col)] = idx
                            found = idx
                            break
                if found is not None:
                    break

    return coord_to_int


def iter_tile_pips(device: Any) -> Iterator[Tuple[Any, str, str, bool]]:
    """
    Iterate over all PIPs (Programmable Interconnect Points) in the device.

    Yields:
        (tile, start_wire, end_wire, forward) for each PIP
    """
    for tile in device.getAllTiles():
        if tile is None:
            continue
        for wire in tile.getWires():
            if wire is None:
                continue
            pips = wire.getBackwardPIPs()
            if pips is None:
                continue
            for pip in pips:
                if pip.isRouteThru():
                    continue
                yield (
                    pip.getTile(),
                    pip.getStartWireName(),
                    pip.getEndWireName(),
                    True,
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load design and inspect routing resources")
    parser.add_argument("--phys", required=True, help="Path to .phys Physical Netlist")
    parser.add_argument("--netlist", help="Path to .netlist Logical Netlist")
    parser.add_argument("--device", default="xcvu3p-ffvc1517-2-e", help="Device name")
    args = parser.parse_args()

    print("Loading design...")
    design = load_design(args.phys, args.netlist)
    print(f"  Design: {design.getName()}")

    nets_to_route = get_nets_to_route(design)
    total_pins = sum(len(pins) for _, pins in nets_to_route)
    print(f"  Nets to route: {len(nets_to_route)}")
    print(f"  total sink pins: {total_pins}")
    print_net_fanout_stats(design)

    print("\nLoading device and routing resources...")
    device = get_device(args.device)
    res = get_routing_resources(device)
    print(f"  Device: {device.getName()}")
    print(f"  Size: {res['device_rows']} x {res['device_cols']}")
    print(f"  Tiles: {len(res['tiles'])}")
