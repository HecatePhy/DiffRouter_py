"""
Visualize FPGA routing as a 2D tile grid with pin-to-pin net connections.

- 2D grid: each cell represents a tile; position from tile.getColumn(), tile.getRow()
- For each net: draw lines between tiles that contain pins of that net
"""

import itertools
from typing import Optional, List, Set, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def _get_net_tiles(net: Any) -> Set[Tuple[int, int]]:
    """Get unique (col, row) tile positions for all pins of a net."""
    tiles = set()
    for pin in net.getPins():
        tile = pin.getTile()
        if tile is not None:
            col = tile.getColumn()
            row = tile.getRow()
            tiles.add((col, row))
    return tiles


def _get_net_pin_locations(net: Any) -> List[Tuple[str, int, int]]:
    """Get (pin_name, col, row) for each pin of a net."""
    result = []
    for pin in net.getPins():
        tile = pin.getTile()
        if tile is not None:
            pin_name = str(pin)  # e.g. wire name or pin descriptor
            result.append((pin_name, tile.getColumn(), tile.getRow()))
    return result


def visualize_routing(
    design: Any,
    device: Any,
    max_nets: Optional[int] = None,
    min_fanout: int = 5,
    draw_grid: bool = True,
    grid_alpha: float = 0.15,
    line_alpha: float = 0.6,
    output_path: Optional[str] = None,
    log_path: Optional[str] = None,
    show_plot: bool = True,
    connect_all_pairs: bool = True,
) -> None:
    """Visualize FPGA routing: 2D tile grid with pin-to-pin lines per net.

    Args:
        design: RapidWright Design (has getNets())
        device: RapidWright Device (has getColumns(), getRows())
        max_nets: If set, only visualize the first N nets (for large designs)
        min_fanout: Only visualize nets with fanout > this (default: 5)
        draw_grid: Whether to draw tile grid lines
        grid_alpha: Alpha for grid lines
        line_alpha: Alpha for net lines
        output_path: Path to save the figure
        log_path: Path to save a log of nets drawn and their pin tile positions
        show_plot: Whether to display interactively
        connect_all_pairs: If True, draw lines between all tile pairs per net;
            if False, draw star (source to each sink) when source exists
    """
    device_cols = device.getColumns()
    device_rows = device.getRows()

    nets = list(design.getNets())

    # Collect (net, tiles) for nets we will draw (have source, 2+ distinct tiles, fanout > min_fanout)
    net_tiles_list: List[Tuple[Any, Set[Tuple[int, int]]]] = []
    for net in nets:
        if net.getSource() is None and not net.isStaticNet():
            continue
        sink_count = len(list(net.getSinkPins()))
        if sink_count <= min_fanout:
            continue
        tiles = _get_net_tiles(net)
        if len(tiles) < 2:
            continue
        net_tiles_list.append((net, tiles))
        if max_nets is not None and len(net_tiles_list) >= max_nets:
            break

    # Write log of nets and pin locations
    if log_path:
        with open(log_path, "w") as f:
            f.write("# Visualizer log: net name -> pin tile positions (col, row)\n")
            f.write(f"# Device: {device.getName()}, {device_rows} x {device_cols}\n")
            f.write(f"# Fanout > {min_fanout}, Nets drawn: {len(net_tiles_list)}\n\n")
            for i, (net, tiles) in enumerate(net_tiles_list):
                net_name = str(net.getName()) if net.getName() else f"<unnamed_{i}>"
                pin_locs = _get_net_pin_locations(net)
                tile_list = sorted(tiles, key=lambda t: (t[1], t[0]))  # row, col
                f.write(f"net[{i}] {net_name}\n")
                f.write(f"  unique tiles (col,row): {tile_list}\n")
                f.write(f"  pins:\n")
                for pin_name, col, row in pin_locs:
                    f.write(f"    {pin_name} -> ({col}, {row})\n")
                f.write("\n")
        print(f"Visualizer log saved to: {log_path}")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw tile grid
    if draw_grid:
        for c in range(0, device_cols + 1):
            ax.axvline(c, color="gray", alpha=grid_alpha, linewidth=0.5)
        for r in range(0, device_rows + 1):
            ax.axhline(r, color="gray", alpha=grid_alpha, linewidth=0.5)

    # Color cycle for nets
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(colors) < len(net_tiles_list):
        cmap = plt.cm.tab20
        colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, max(len(net_tiles_list), 20))]

    # Draw lines for each net
    for i, (net, tiles) in enumerate(net_tiles_list):
        color = colors[i % len(colors)]
        tile_list = list(tiles)
        if connect_all_pairs:
            for (c1, r1), (c2, r2) in itertools.combinations(tile_list, 2):
                ax.plot([c1, c2], [r1, r2], color=color, alpha=line_alpha, linewidth=1.0)
        else:
            # Star: pick first tile as "center", connect to others
            if len(tile_list) < 2:
                continue
            c0, r0 = tile_list[0]
            for c, r in tile_list[1:]:
                ax.plot([c0, c], [r0, r], color=color, alpha=line_alpha, linewidth=1.0)

    ax.set_xlim(-0.5, device_cols)
    ax.set_ylim(-0.5, device_rows)
    ax.set_aspect("equal")
    ax.set_xlabel("Column (X)")
    ax.set_ylabel("Row (Y)")
    ax.set_title(f"FPGA Routing: {len(net_tiles_list)} nets, Device {device.getName()}")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_density_map(
    design: Any,
    device: Any,
    min_fanout: int = 5,
    output_path: Optional[str] = None,
    show_plot: bool = True,
    colormap: str = "YlOrRd",
) -> np.ndarray:
    """Visualize net bounding-box density map.

    For each net with fanout > min_fanout:
    - Compute bounding box of pin tiles
    - For each grid cell inside the bbox, add fanout/bbox_area to density

    Returns:
        density: 2D array (rows x cols) of accumulated density values
    """
    device_cols = device.getColumns()
    device_rows = device.getRows()
    density = np.zeros((device_rows, device_cols), dtype=np.float64)

    nets = list(design.getNets())
    for net in nets:
        if net.getSource() is None and not net.isStaticNet():
            continue
        sink_count = len(list(net.getSinkPins()))
        if sink_count <= min_fanout:
            continue
        tiles = _get_net_tiles(net)
        if len(tiles) < 2:
            continue

        # Bounding box: (min_col, max_col, min_row, max_row)
        cols = [t[0] for t in tiles]
        rows = [t[1] for t in tiles]
        min_col, max_col = min(cols), max(cols)
        min_row, max_row = min(rows), max(rows)

        bbox_area = (max_col - min_col + 1) * (max_row - min_row + 1)
        contrib = sink_count / bbox_area if bbox_area > 0 else 0.0
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if 0 <= r < device_rows and 0 <= c < device_cols:
                    density[r, c] += contrib

    fig, ax = plt.subplots(figsize=(14, 10))
    try:
        cmap = plt.colormaps[colormap]
    except (KeyError, AttributeError):
        cmap = getattr(plt.cm, colormap, plt.cm.YlOrRd)
    im = ax.imshow(density, cmap=cmap, aspect="auto", interpolation="nearest", origin="lower")
    plt.colorbar(im, ax=ax, label="Density (sum of fanout/bbox_area over nets whose bbox contains tile)")
    ax.set_xlabel("Column (X)")
    ax.set_ylabel("Row (Y)")
    ax.set_title(f"Net Bounding-Box Density (fanout > {min_fanout}), Device {device.getName()}")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Density map saved to: {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()

    return density


def render_congestion_frame(
    congestion_grid: np.ndarray,
    title: str = "Congestion (flow/capacity)",
    colormap: str = "YlOrRd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 100,
) -> np.ndarray:
    """Render congestion map to RGB array (H, W, 3) uint8 for GIF frames. No file I/O."""
    import io
    from PIL import Image

    vmin = vmin if vmin is not None else 0.0
    vmax = vmax if vmax is not None else (float(np.nanmax(congestion_grid)) if congestion_grid.size > 0 else 1.0)
    try:
        cmap = plt.colormaps[colormap]
    except (KeyError, AttributeError):
        cmap = getattr(plt.cm, colormap, plt.cm.YlOrRd)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(
        congestion_grid,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="Congestion (flow / capacity)")
    ax.set_xlabel("Column (X)")
    ax.set_ylabel("Row (Y)")
    ax.set_title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img)


def visualize_congestion_map(
    congestion_grid: np.ndarray,
    title: str = "Congestion (flow/capacity)",
    output_path: Optional[str] = None,
    show_plot: bool = False,
    colormap: str = "YlOrRd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Visualize congestion map: flow/capacity per tile (2D heatmap).

    Args:
        congestion_grid: 2D array (rows x cols), value = max(flow/capacity) at tile
        title: Plot title
        output_path: Path to save figure
        show_plot: Whether to display interactively
        colormap: Matplotlib colormap name
        vmin, vmax: Color scale limits (default: 0 to max of grid)
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    try:
        cmap = plt.colormaps[colormap]
    except (KeyError, AttributeError):
        cmap = getattr(plt.cm, colormap, plt.cm.YlOrRd)

    vmin = vmin if vmin is not None else 0.0
    vmax = vmax if vmax is not None else float(np.nanmax(congestion_grid)) if congestion_grid.size > 0 else 1.0

    im = ax.imshow(
        congestion_grid,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="Congestion (flow / capacity)")
    ax.set_xlabel("Column (X)")
    ax.set_ylabel("Row (Y)")
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Congestion map saved to: {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_density_map_from_router(
    router: Any,
    min_fanout: int = 5,
    output_path: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper: density map from a Router instance."""
    if router.design is None:
        raise ValueError("Router design must be loaded before visualization")
    return visualize_density_map(
        router.design,
        router.device,
        min_fanout=min_fanout,
        output_path=output_path,
        **kwargs,
    )


def visualize_routing_from_router(
    router: Any,
    max_nets: Optional[int] = None,
    min_fanout: int = 5,
    output_path: Optional[str] = None,
    log_path: Optional[str] = None,
    **kwargs,
) -> None:
    """Convenience wrapper: visualize from a Router instance."""
    if router.design is None:
        raise ValueError("Router design must be loaded before visualization")
    visualize_routing(
        router.design,
        router.device,
        max_nets=max_nets,
        min_fanout=min_fanout,
        output_path=output_path,
        log_path=log_path,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    import os
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _root)
    os.chdir(_root)
    from src.load_design import load_design, get_device

    parser = argparse.ArgumentParser(description="Visualize FPGA routing (nets with fanout > 5)")
    parser.add_argument("--max-nets", type=int, default=None, help="Max nets to draw (default: all)")
    parser.add_argument("--density", action="store_true", help="Show density map instead of net lines")
    parser.add_argument("--testcase", default="boom_soc_v2", help="Testcase name")
    parser.add_argument("--data", default="./data/", help="Data directory")
    args = parser.parse_args()

    data_prefix = args.data
    testcase = args.testcase
    device_name = "xcvu3p-ffvc1517-2-e"
    netlist_path = os.path.join(data_prefix, testcase, f"{testcase}.netlist")
    physical_path = os.path.join(data_prefix, testcase, f"{testcase}_unrouted.phys")

    design = load_design(physical_path, netlist_path)
    device = get_device(device_name)

    if args.density:
        output_file = f"density_{testcase}.png"
        visualize_density_map(
            design,
            device,
            output_path=output_file,
            show_plot=False,
        )
    else:
        output_file = f"routing_{testcase}.png"
        log_file = f"routing_{testcase}.log"
        visualize_routing(
            design,
            device,
            max_nets=args.max_nets,
            output_path=output_file,
            log_path=log_file,
            show_plot=False,
        )
