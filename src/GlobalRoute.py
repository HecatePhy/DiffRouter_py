"""
Differentiable Global Router using PyTorch.

CLI entry point. Core implementation in src/router/ and src/rrg/.
"""

from typing import List, Optional, Tuple

from src.rrg.rrg import RRG
from src.router.global_router import GlobalRouter, get_net_tiles, plot_bbox_distribution

_get_net_tiles = get_net_tiles
_plot_bbox_distribution = plot_bbox_distribution


def write_rrg_capacity_log(
    device_name: Optional[str] = None,
    output_path: str = "rrg_capacity.log",
    rrg_path: Optional[str] = None,
) -> None:
    from src.load_design import get_device, get_tile_graph, get_tile_edge_capacities, load_rrg

    if rrg_path:
        tile_graph, edge_capacities, _, _ = load_rrg(rrg_path)
    elif device_name:
        device = get_device(device_name)
        tile_graph = get_tile_graph(device)
        edge_capacities, wire_edges = get_tile_edge_capacities(device, tile_graph)
        tile_graph = dict(tile_graph)
        tile_graph["edges"] = sorted(set(tile_graph["edges"]) | wire_edges)
    else:
        raise ValueError("Either device_name or rrg_path must be provided")
    rrg = RRG(tile_graph, edge_capacities, edge_mode="directed")
    rrg.write_capacity_log(output_path)


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _root)
    os.chdir(_root)
    from src.load_design import load_design, get_device
    from src.Timer import Timer

    parser = argparse.ArgumentParser(description="Differentiable Global Router")
    parser.add_argument("--testcase", default="boom_soc_v2", help="Testcase name")
    parser.add_argument("--data", default="./data/", help="Data directory")
    parser.add_argument("--rrg-log", default="rrg_capacity.log", help="Path for directed RRG capacity log")
    parser.add_argument("--rrg-only", action="store_true", help="Only output RRG capacity log (no design load)")
    parser.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt", help="Path to RRG JSON/PT file")
    parser.add_argument("--device", default="xcvu3p-ffvc1517-2-e", help="Device name")
    parser.add_argument("--from-device", action="store_true", help="Build RRG from RapidWright device")
    parser.add_argument("--viz-dir", default="results/checkpoint", help="Save congestion evolution GIF")
    parser.add_argument("--max-iterations", type=int, default=1000, help="Max optimization iterations")
    parser.add_argument("--viz-interval", type=int, default=50, help="Dump congestion viz every N iters")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--bbox-only", action="store_true", help="Output bbox distribution and exit")
    parser.add_argument("--bbox-fast", action="store_true", help="With --bbox-only: skip RRG load")
    parser.add_argument("--bbox-out", default=None, help="Save per-net bbox to JSON")
    parser.add_argument("--bbox-plot", default=None, help="Save bbox distribution plot PNG")
    parser.add_argument("--quiet", "-q", action="store_true", help="Run optimization only with logs")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint .pt")
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N iters")
    parser.add_argument("--connectivity-solver", choices=["solve", "cg"], default="cg")
    parser.add_argument("--conn-net-batch", type=int, default=0)
    parser.add_argument("--conn-col-chunk", type=int, default=32)
    parser.add_argument("--conn-cg-max-iter", type=int, default=100)
    parser.add_argument("--conn-edge-chunk", type=int, default=0,
                        help="Bound connectivity matvec temporary to [edge_chunk, col_chunk] "
                             "rows; 0=off. Use ~8000000 for large designs to avoid OOM.")
    parser.add_argument("--flow-net-batch", type=int, default=0)
    parser.add_argument("--edge-scope", choices=["bbox", "corridor"], default="corridor")
    parser.add_argument("--corridor-width", type=int, default=2)
    parser.add_argument("--max-edges-per-net", type=int, default=50000)
    parser.add_argument("--lr-x", type=float, default=0.01)
    parser.add_argument("--lr-lam", type=float, default=0.1)
    parser.add_argument("--w-wl", type=float, default=1.0)
    parser.add_argument("--w-conn", type=float, default=1.0)
    parser.add_argument("--w-flow", type=float, default=1.0)
    parser.add_argument("--edge-mode", choices=["directed", "undirected"], default="directed",
                        help="Edge variables: directed arcs or undirected phys edges (no flow term)")
    parser.add_argument("--net-index", default=None, help="Path to pre-built net index .pt")
    parser.add_argument("--live-build-nets", action="store_true",
                        help="Debug: build net list from design (requires Java load)")
    args = parser.parse_args()

    import torch

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    quiet = args.quiet
    w_flow = 0.0 if args.edge_mode == "undirected" else args.w_flow
    if args.edge_mode == "undirected" and args.w_flow != 0.0 and not quiet:
        print("  Note: --w-flow ignored in undirected edge mode")

    if not quiet:
        print("[Step 0] Device setup")
        if device.type == "cuda":
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  Using CPU")

    if args.rrg_only:
        print("[Step 1] RRG capacity log only")
        with Timer("rrg_log", logger=print):
            if args.rrg:
                write_rrg_capacity_log(rrg_path=args.rrg, output_path=args.rrg_log)
            else:
                write_rrg_capacity_log(device_name=args.device, output_path=args.rrg_log)
    else:
        data_prefix = args.data
        testcase = args.testcase
        netlist_path = os.path.join(data_prefix, testcase, f"{testcase}.netlist")
        physical_path = os.path.join(data_prefix, testcase, f"{testcase}_unrouted.phys")

        from src.router.net_index import default_net_index_path, require_net_index_path

        net_index_path = args.net_index or default_net_index_path(
            data_prefix, testcase, args.rrg, args.edge_mode, 0, 0.1,
            route_filter="stubs",
            edge_scope=args.edge_scope,
            corridor_width=args.corridor_width,
            max_edges_per_net=args.max_edges_per_net,
        )
        cap = f" --max-edges-per-net {args.max_edges_per_net}" if args.max_edges_per_net else ""
        prebuild_hint = (
            f"python scripts/PrebuildNetIndex.py --testcase {testcase} "
            f"--rrg {args.rrg} --edge-mode {args.edge_mode} "
            f"--edge-scope {args.edge_scope} --corridor-width {args.corridor_width}{cap}"
        )

        if args.bbox_only and args.bbox_fast:
            print("[Step 1] Loading design (bbox-fast)")
            with Timer("load_design", logger=print):
                design = load_design(physical_path, netlist_path)
            print("[Step 2] Bbox from pin positions (no RRG load)")
            dev_obj = get_device(args.device)
            device_cols = dev_obj.getColumns()
            device_rows = dev_obj.getRows()
            min_fanout = 5
            expansion_ratio = 0.1
            bboxes = []
            for net in design.getNets():
                if net.getSource() is None or net.isStaticNet():
                    continue
                sink_pins = list(net.getSinkPins())
                if len(sink_pins) <= min_fanout:
                    continue
                tiles = _get_net_tiles(net)
                if len(tiles) < 2:
                    continue
                cols, rows = [t[0] for t in tiles], [t[1] for t in tiles]
                min_col, max_col = min(cols), max(cols)
                min_row, max_row = min(rows), max(rows)
                w, h = max(1, max_col - min_col), max(1, max_row - min_row)
                exp_w = max(1, int(w * expansion_ratio))
                exp_h = max(1, int(h * expansion_ratio))
                min_col = max(0, min_col - exp_w)
                max_col = min(device_cols, max_col + exp_w)
                min_row = max(0, min_row - exp_h)
                max_row = min(device_rows, max_row + exp_h)
                bboxes.append((net, min_col, max_col, min_row, max_row,
                               max_col - min_col + 1, max_row - min_row + 1))
            widths = sorted([b[5] for b in bboxes])
            heights = sorted([b[6] for b in bboxes])
            areas = sorted([b[5] * b[6] for b in bboxes])
            n = len(bboxes)
            if n > 0:
                print("Bounding box size (tile_col x tile_row, no RRG):")
                print(f"  tile_col: min={widths[0]} max={widths[-1]} median={widths[n//2]:.0f} mean={sum(widths)/n:.1f}")
                print(f"  tile_row: min={heights[0]} max={heights[-1]} median={heights[n//2]:.0f} mean={sum(heights)/n:.1f}")
                print(f"  area:     min={areas[0]} max={areas[-1]} median={areas[n//2]:.0f} mean={sum(areas)/n:.1f}")
            if args.bbox_out and bboxes:
                out = [{"net_idx": i, "net_name": str(net.getName()) or f"net_{i}",
                        "min_col": mc, "max_col": Mx, "min_row": mr, "max_row": My,
                        "tile_col": tc, "tile_row": tr, "area": tc * tr}
                       for i, (net, mc, Mx, mr, My, tc, tr) in enumerate(bboxes)]
                with open(args.bbox_out, "w") as f:
                    json.dump(out, f, indent=2)
            plot_path = args.bbox_plot or "results/checkpoint/bbox_distribution.png"
            if plot_path and bboxes:
                _plot_bbox_distribution([b[5] for b in bboxes], [b[6] for b in bboxes],
                                        [b[5] * b[6] for b in bboxes], output_path=plot_path)
            print("[Done] (bbox-only, fast mode)")
            sys.exit(0)

        if not quiet:
            print("[Step 2] Building GlobalRouter")
        else:
            print("Building router...")
        rrg_log = None if quiet else args.rrg_log
        with Timer("build_router", logger=print):
            if args.from_device:
                if not quiet:
                    print("[Step 1] Loading design (from-device path)")
                with Timer("load_design", logger=print):
                    design = load_design(physical_path, netlist_path)
                dev_obj = get_device(args.device)
                router = GlobalRouter.from_device(
                    design,
                    dev_obj,
                    min_fanout=0,
                    rrg_log_path=rrg_log,
                    device=device,
                    edge_mode=args.edge_mode,
                    verbose=not quiet,
                )
            elif args.live_build_nets:
                if not quiet:
                    print("[Step 1] Loading design (live-build-nets)")
                with Timer("load_design", logger=print):
                    design = load_design(physical_path, netlist_path)
                router = GlobalRouter.load_live(
                    design,
                    args.rrg,
                    device=device,
                    edge_mode=args.edge_mode,
                    verbose=not quiet,
                )
            else:
                require_net_index_path(net_index_path, prebuild_hint)
                router = GlobalRouter.load(
                    args.rrg,
                    net_index_path,
                    device=device,
                    edge_mode=args.edge_mode,
                    verbose=not quiet,
                )
        router.conn_col_chunk = args.conn_col_chunk
        router.conn_cg_max_iter = args.conn_cg_max_iter
        router.conn_edge_chunk = args.conn_edge_chunk
        if not quiet:
            router.print_info()
            router.print_bbox_size_distribution()

        if args.bbox_only:
            tile_cols, tile_rows, areas, int_tiles_list = [], [], [], []
            for bbox in router.net_bbox:
                min_col, max_col, min_row, max_row = bbox
                w, h = max_col - min_col + 1, max_row - min_row + 1
                tile_cols.append(w)
                tile_rows.append(h)
                areas.append(w * h)
                int_tiles_list.append(router._count_int_tiles_in_bbox(min_col, max_col, min_row, max_row))
            if args.bbox_out:
                bbox_data = []
                for i, bbox in enumerate(router.net_bbox):
                    min_col, max_col, min_row, max_row = bbox
                    w, h = max_col - min_col + 1, max_row - min_row + 1
                    net_name = router.net_names[i] if i < len(router.net_names) else f"net_{i}"
                    bbox_data.append({
                        "net_idx": i, "net_name": net_name,
                        "min_col": min_col, "max_col": max_col,
                        "min_row": min_row, "max_row": max_row,
                        "tile_col": w, "tile_row": h, "area": w * h,
                        "int_tiles": int_tiles_list[i],
                    })
                with open(args.bbox_out, "w") as f:
                    json.dump(bbox_data, f, indent=2)
            plot_path = args.bbox_plot or "results/checkpoint/bbox_distribution.png"
            if plot_path and tile_cols:
                _plot_bbox_distribution(tile_cols, tile_rows, areas,
                                        int_tiles=int_tiles_list, output_path=plot_path)
            print("[Done] (bbox-only)")
            sys.exit(0)

        checkpoint_dir = args.viz_dir or "results/checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)

        if not quiet:
            print("[Step 3] Initializing variables")
        x = router.init_variables()
        if not quiet:
            loss = router.total_loss(x, w_wl=args.w_wl, w_cong=10.0, w_conn=args.w_conn,
                                     w_flow=w_flow,
                                     connectivity_solver=args.connectivity_solver)
            print(f"  Initial loss: {loss.item():.4f}")
            print("[Step 4] Running Augmented Lagrangian optimization")
        else:
            print("Optimizing...")

        with Timer("optimize", logger=print):
            x_opt = router.optimize_augmented_lagrangian(
                x=x,
                num_outer=max(1, args.max_iterations // 5),
                num_inner=5,
                max_iterations=args.max_iterations,
                viz_interval=args.viz_interval,
                viz_dir=checkpoint_dir,
                checkpoint_dir=checkpoint_dir,
                save_every=args.save_every,
                resume_path=args.resume,
                lr_x=args.lr_x,
                lr_lam=args.lr_lam,
                w_wl=args.w_wl,
                w_conn=args.w_conn,
                w_flow=w_flow,
                connectivity_solver=args.connectivity_solver,
                conn_net_batch=args.conn_net_batch,
                flow_net_batch=args.flow_net_batch,
                verbose=True,
                log_setup=not quiet,
            )
        final_loss = router.total_loss(x_opt, w_wl=args.w_wl, w_cong=10.0, w_conn=args.w_conn,
                                       w_flow=w_flow,
                                       connectivity_solver=args.connectivity_solver, conn_net_batch=0)
        print(f"  Final loss: {final_loss.item():.4f}")
        if not quiet:
            print("[Done]")
