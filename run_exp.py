"""End-to-end DiffRouter pipeline: global optimize → extract → detailed route → write."""

import argparse
import json
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
os.chdir(_root)

import torch

from src.Timer import Timer
from src.detailed_route import load_design_for_routing, route_design_detailed
from src.io.write_design import ensure_result_dir, write_routed_dcp, write_routed_phys
from src.load_design import load_design
from src.router.global_router import GlobalRouter
from src.router.route_extractor import RouteExtractor


def _net_index_path(args) -> str:
    from src.router.net_index import default_net_index_path

    if args.net_index:
        return args.net_index
    return default_net_index_path(
        args.data,
        args.testcase,
        args.rrg,
        args.edge_mode,
        args.min_fanout,
        0.1,
        route_filter="stubs",
        edge_scope=args.edge_scope,
        corridor_width=args.corridor_width,
        max_edges_per_net=args.max_edges_per_net,
    )


def _prebuild_hint(args) -> str:
    cap = f" --max-edges-per-net {args.max_edges_per_net}" if args.max_edges_per_net else ""
    return (
        f"python scripts/PrebuildNetIndex.py --testcase {args.testcase} "
        f"--rrg {args.rrg} --edge-mode {args.edge_mode} "
        f"--min-fanout {args.min_fanout} --edge-scope {args.edge_scope} "
        f"--corridor-width {args.corridor_width}{cap}"
    )


def _load_router(args, device):
    if args.live_build_nets:
        netlist_path = os.path.join(args.data, args.testcase, f"{args.testcase}.netlist")
        physical_path = os.path.join(args.data, args.testcase, f"{args.testcase}_unrouted.phys")
        with Timer("load_design", logger=print):
            design = load_design(physical_path, netlist_path)
        with Timer("build_router", logger=print):
            router = GlobalRouter.load_live(
                design,
                args.rrg,
                device=device,
                edge_mode=args.edge_mode,
                min_fanout=args.min_fanout,
                max_nets=args.max_nets,
                verbose=not args.quiet,
            )
        router.conn_col_chunk = getattr(args, "conn_col_chunk", 32)
        router.conn_cg_max_iter = getattr(args, "conn_cg_max_iter", 100)
        router.conn_edge_chunk = getattr(args, "conn_edge_chunk", 0)
        router.conn_warm_start = getattr(args, "conn_warm_start", True)
        return router
    from src.router.net_index import require_net_index_path

    net_index = require_net_index_path(_net_index_path(args), _prebuild_hint(args))
    with Timer("build_router", logger=print):
        router = GlobalRouter.load(
            args.rrg,
            net_index,
            device=device,
            edge_mode=args.edge_mode,
            verbose=not args.quiet,
        )
    router.conn_col_chunk = getattr(args, "conn_col_chunk", 32)
    router.conn_cg_max_iter = getattr(args, "conn_cg_max_iter", 100)
    router.conn_edge_chunk = getattr(args, "conn_edge_chunk", 0)
    router.conn_warm_start = getattr(args, "conn_warm_start", True)
    return router


def run_pipeline(args) -> dict:
    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data_prefix = args.data
    testcase = args.testcase
    netlist_path = os.path.join(data_prefix, testcase, f"{testcase}.netlist")
    physical_path = os.path.join(data_prefix, testcase, f"{testcase}_unrouted.phys")
    result_dir = ensure_result_dir(testcase, args.results)
    checkpoint_dir = os.path.join(result_dir, "checkpoint")

    if not args.quiet:
        print(f"[Pipeline] testcase={testcase} device={device}")
    else:
        print(f"Pipeline: {testcase}")

    metrics = {"testcase": testcase, "device": str(device), "edge_mode": args.edge_mode}

    w_flow = 0.0 if args.edge_mode == "undirected" else args.w_flow
    if args.edge_mode == "undirected" and args.w_flow != 0.0 and not args.quiet:
        print("  Note: --w-flow ignored in undirected edge mode")

    # --- Global routing ---
    if args.skip_global and os.path.isfile(os.path.join(checkpoint_dir, "global_x.pt")):
        print("Loading cached global_x.pt (skip global)...")
        router = _load_router(args, device)
        x_opt = torch.load(
            os.path.join(checkpoint_dir, "global_x.pt"),
            map_location=device,
        )
    else:
        router = _load_router(args, device)
        if not args.quiet:
            router.print_info()

        x_init = None
        resume = args.resume or (
            os.path.join(checkpoint_dir, "checkpoint_final.pt")
            if args.auto_resume
            else None
        )
        if resume and os.path.isfile(resume):
            print(f"Resuming global optimization from {resume}")

        with Timer("optimize", logger=print):
            x_opt = router.optimize_augmented_lagrangian(
                x=x_init,
                num_outer=max(1, args.max_iterations // args.num_inner),
                num_inner=args.num_inner,
                max_iterations=args.max_iterations,
                viz_interval=args.viz_interval,
                viz_dir=checkpoint_dir if args.viz else None,
                checkpoint_dir=checkpoint_dir,
                save_every=args.save_every,
                resume_path=resume if resume and os.path.isfile(resume) else None,
                lr_x=args.lr_x,
                lr_lam=args.lr_lam,
                rho=args.rho,
                w_wl=args.w_wl,
                w_conn=args.w_conn,
                w_flow=w_flow,
                w_disc=args.w_disc,
                disc_ramp_outer=args.disc_ramp_outer,
                connectivity_solver=args.connectivity_solver,
                conn_net_batch=args.conn_net_batch,
                flow_net_batch=args.flow_net_batch,
                verbose=True,
                log_setup=not args.quiet,
            )

        final_loss = router.total_loss(
            x_opt,
            w_wl=args.w_wl,
            w_cong=10.0,
            w_conn=args.w_conn,
            w_flow=w_flow,
            connectivity_solver=args.connectivity_solver,
            conn_net_batch=0,
        )
        metrics["global_final_loss"] = final_loss.item()
        print(f"  Global final loss: {final_loss.item():.4f}")

    # --- Route extraction ---
    with Timer("extract", logger=print):
        cong_map = router.get_congestion_map(x_opt)
        extractor = RouteExtractor(threshold=args.route_threshold)
        paths = extractor.extract(router, x_opt, cong_map)
        connected = sum(
            1
            for i in range(router.num_nets)
            if router.net_src_tile[i] in paths.get(i, [])
            and all(s in paths.get(i, []) for s in router.net_sink_tiles[i])
        )
        metrics["extracted_nets"] = len(paths)
        metrics["connected_nets"] = connected
        print(f"  Extracted paths: {len(paths)}, connected: {connected}/{router.num_nets}")

        paths_json = os.path.join(result_dir, "tile_paths.json")
        serializable = {
            str(k): v for k, v in paths.items()
        }
        with open(paths_json, "w") as f:
            json.dump(serializable, f, indent=2)

    if args.global_only:
        metrics_path = os.path.join(result_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Done] global-only, metrics: {metrics_path}")
        return metrics

    # --- Detailed routing ---
    with Timer("detailed_route", logger=print):
        design_rw = load_design_for_routing(physical_path, netlist_path)
        router.attach_design(design_rw)
        detail_stats = route_design_detailed(
            design_rw,
            router,
            router.nets,
            paths=paths,
            soft_preserve=args.soft_preserve,
            verbose=not args.quiet,
        )
        metrics.update(detail_stats)

    # --- Write outputs ---
    with Timer("write", logger=print):
        phys_out = os.path.join(result_dir, f"{testcase}_routed.phys")
        dcp_out = os.path.join(result_dir, f"{testcase}_routed.dcp")
        write_routed_phys(design_rw, phys_out)
        write_routed_dcp(design_rw, dcp_out)
        metrics["phys_out"] = phys_out
        metrics["dcp_out"] = dcp_out

    metrics_path = os.path.join(result_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Done] metrics: {metrics_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="DiffRouter end-to-end pipeline")
    parser.add_argument("--testcase", default="boom_soc_v2")
    parser.add_argument("--data", default="./data/")
    parser.add_argument("--results", default="./results/")
    parser.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--num-inner", type=int, default=5)
    parser.add_argument("--viz-interval", type=int, default=50)
    parser.add_argument("--viz", action="store_true", help="Save congestion GIF")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--min-fanout", type=int, default=0)
    parser.add_argument("--max-nets", type=int, default=None, help="Limit nets for quick tests")
    parser.add_argument("--lr-x", type=float, default=0.01)
    parser.add_argument("--lr-lam", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--w-wl", type=float, default=1.0)
    parser.add_argument("--w-conn", type=float, default=1.0)
    parser.add_argument("--w-disc", type=float, default=0.0,
                        help="Discretization penalty weight (sum x*(1-x)); drives x to {0,1}")
    parser.add_argument("--disc-ramp-outer", type=int, default=0,
                        help="Anneal w_disc from 0 to full over this many outer iters (0=constant)")
    parser.add_argument("--w-flow", type=float, default=1.0)
    parser.add_argument("--edge-mode", choices=["directed", "undirected"], default="directed",
                        help="Edge variables: directed arcs or undirected phys edges (no flow term)")
    parser.add_argument("--net-index", default=None, help="Path to pre-built net index .pt")
    parser.add_argument("--live-build-nets", action="store_true",
                        help="Debug: build net list from design (requires Java load)")
    parser.add_argument("--route-threshold", type=float, default=0.01,
                        help="Min edge weight for route extraction")
    parser.add_argument("--connectivity-solver", choices=["solve", "cg", "grouped"], default="cg",
                        help="grouped = net-grouped subgraph CG (fast + low memory at scale)")
    parser.add_argument("--conn-warm-start", action="store_true", default=True,
                        help="grouped solver: warm-start CG from previous iter's solution")
    parser.add_argument("--no-conn-warm-start", dest="conn_warm_start", action="store_false")
    parser.add_argument("--conn-net-batch", type=int, default=0,
                        help="Legacy: per-net solve subsample (ignored when solver=cg)")
    parser.add_argument("--conn-edge-chunk", type=int, default=0,
                        help="Bound connectivity Laplacian matvec temporary to "
                             "[edge_chunk, col_chunk] rows; 0 = no chunking. "
                             "Set ~8000000 for large designs to avoid OOM.")
    parser.add_argument("--conn-col-chunk", type=int, default=32,
                        help="Sink columns per CG chunk (lower = less GPU memory)")
    parser.add_argument("--conn-cg-max-iter", type=int, default=100)
    parser.add_argument("--edge-scope", choices=["bbox", "corridor"], default="corridor",
                        help="Net edge region: full bbox or L-shaped src->sink corridors")
    parser.add_argument("--corridor-width", type=int, default=2,
                        help="Half-width (tiles) of L-shaped routing corridors")
    parser.add_argument("--max-edges-per-net", type=int, default=50000,
                        help="Skip net if corridor cannot fit within this edge cap (0=unlimited)")
    parser.add_argument("--flow-net-batch", type=int, default=0,
                        help="Sample N nets per iter for flow penalty (0=all)")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume global opt")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--skip-global", action="store_true")
    parser.add_argument("--global-only", action="store_true")
    parser.add_argument("--soft-preserve", action="store_true")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
