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
        router.conn_edge_chunk = getattr(args, "conn_edge_chunk", 8_000_000)
        router.conn_warm_start = getattr(args, "conn_warm_start", True)
        router.conn_max_sinks = getattr(args, "conn_max_sinks", 0)
        router.conn_super_sink = getattr(args, "conn_super_sink", False)
        router.conn_sat_alpha = getattr(args, "conn_sat_alpha", 0.0)
        router.flow_demand_mode = getattr(args, "flow_demand", "fanout")
        router.congestion_mode = getattr(args, "congestion_mode", "hard")
        router.congestion_tau = getattr(args, "congestion_tau", 0.1)
        router.init_mode = getattr(args, "init_mode", "uniform")
        router.init_on_path = getattr(args, "init_on_path", 1.0)
        router.init_off_path = getattr(args, "init_off_path", 0.0)
        _ng = getattr(args, "conn_multi_gpu", 1)
        router.conn_mg_devices = (
            [torch.device("cuda:%d" % i) for i in range(_ng)] if _ng > 1 else None)
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
    router.conn_edge_chunk = getattr(args, "conn_edge_chunk", 8_000_000)
    router.conn_warm_start = getattr(args, "conn_warm_start", True)
    router.conn_max_sinks = getattr(args, "conn_max_sinks", 0)
    router.conn_super_sink = getattr(args, "conn_super_sink", False)
    router.conn_sat_alpha = getattr(args, "conn_sat_alpha", 0.0)
    router.flow_demand_mode = getattr(args, "flow_demand", "fanout")
    router.congestion_mode = getattr(args, "congestion_mode", "hard")
    router.congestion_tau = getattr(args, "congestion_tau", 0.1)
    router.init_mode = getattr(args, "init_mode", "uniform")
    router.init_on_path = getattr(args, "init_on_path", 1.0)
    router.init_off_path = getattr(args, "init_off_path", 0.0)
    _ng = getattr(args, "conn_multi_gpu", 1)
    router.conn_mg_devices = (
        [torch.device("cuda:%d" % i) for i in range(_ng)] if _ng > 1 else None)
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
                conn_every=args.conn_every,
                grad_balance=args.grad_balance,
                balance_ratio=args.balance_ratio,
                balance_every=args.balance_every,
                optimizer_kind=args.optimizer_kind,
                eg_lr=args.eg_lr,
                eg_clip=args.eg_clip,
                lam_update=args.lam_update,
                lam_mult_eta=args.lam_mult_eta,
                lam_base=args.lam_base,
                conn_freeze_outer=args.conn_freeze_outer,
                early_stop_tol=args.early_stop_tol,
                early_stop_patience=args.early_stop_patience,
                overflow_stop_frac=args.overflow_stop_frac,
                connectivity_solver=args.connectivity_solver,
                conn_net_batch=args.conn_net_batch,
                flow_net_batch=args.flow_net_batch,
                verbose=True,
                log_setup=not args.quiet,
            )

        if args.final_loss:
            # A full connectivity solve over every column, just to report a number.
            # Off by default: on a large design it costs a whole extra solve (and its
            # peak memory) after the optimisation has already finished.
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

    # The AL warm-start cache holds one CG solution per group (several GB on a large
    # design) and is useless once the optimisation is done -- free it before the
    # guide/extraction stages, which need their own working memory.
    router.free_solver_caches(verbose=not args.quiet)

    # --- Potter GR guide (inline: reuses the loaded router, no reload/round-trip) ---
    if args.guide_out:
        from src.router.gpu_guide import export_guide

        with Timer("guide", logger=print):
            metrics["guide_nets"] = export_guide(router, x_opt, args.guide_out)
            metrics["guide_out"] = args.guide_out

    # --- Route extraction ---
    if args.skip_extract:
        print("  Skipping route extraction (--skip-extract)")
        metrics_path = os.path.join(result_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Done] metrics: {metrics_path}")
        return metrics

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
    parser.add_argument("--guide-out", default="",
                        help="Write a Potter GR route guide here, inline after global "
                             "routing (GPU batched shortest paths; reuses the loaded "
                             "router, so no reload round-trip)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip tile-path extraction (not needed when using --guide-out)")
    parser.add_argument("--final-loss", action="store_true",
                        help="Report the full final loss after optimisation. Costs an extra "
                             "full connectivity solve (all columns) purely for reporting.")
    parser.add_argument("--connectivity-solver", choices=["solve", "cg", "grouped"], default="grouped",
                        help="grouped = net-grouped subgraph CG: each chunk solves only its "
                             "nets' subgraph. Default -- far faster and lower-memory at scale. "
                             "'cg' (batched, whole-graph matvec per chunk) and 'solve' (dense "
                             "per-net) are kept for reference and OOM on large designs unless "
                             "--conn-edge-chunk is set.")
    parser.add_argument("--conn-warm-start", action="store_true", default=True,
                        help="grouped solver: warm-start CG from previous iter's solution")
    parser.add_argument("--no-conn-warm-start", dest="conn_warm_start", action="store_false")
    parser.add_argument("--conn-net-batch", type=int, default=0,
                        help="Legacy: per-net solve subsample (ignored when solver=cg)")
    parser.add_argument("--conn-edge-chunk", type=int, default=8_000_000,
                        help="solver=cg only: bound the Laplacian matvec temporary to "
                             "[edge_chunk, col_chunk] rows (0 = unbounded). The unbounded "
                             "temporary is num_vars x col_chunk, which is tens of GB on large "
                             "designs and OOMs; chunking is exact, just accumulated in slices.")
    parser.add_argument("--conn-col-chunk", type=int, default=128,
                        help="Sink columns per CG chunk. Larger amortizes kernel-launch "
                             "overhead (the grouped solver is launch-bound); lower = less "
                             "GPU memory.")
    parser.add_argument("--conn-cg-max-iter", type=int, default=8,
                        help="CG iterations per connectivity solve. 8 suffices with "
                             "--conn-warm-start (each solve starts from the previous "
                             "iteration's solution). Raise if you disable warm-start.")
    parser.add_argument("--conn-max-sinks", type=int, default=0,
                        help="Cap connectivity to first K sinks/net (0=all). Cheap runtime "
                             "lever: huge-fanout nets dominate connectivity cost.")
    parser.add_argument("--init-mode", choices=["uniform", "shortest_path"],
                        default="uniform",
                        help="Initial x. 'uniform' (default) spreads 0.1 over each net's "
                             "corridor, so x~5e-4: usage is then far below capacity, the "
                             "congestion relu has zero gradient, and dR/dw~1/w^2 makes "
                             "effective resistance ~1e8x every other term. "
                             "'shortest_path' starts from the min-hop routing instead, so "
                             "conductances are O(1) and usage is a real net count.")
    parser.add_argument("--init-on-path", type=float, default=1.0,
                        help="--init-mode shortest_path: x on the chosen path edges.")
    parser.add_argument("--init-off-path", type=float, default=0.0,
                        help="--init-mode shortest_path: x on all other corridor edges.")
    parser.add_argument("--optimizer-kind", choices=["adam", "eg"], default="adam",
                        help="Inner update. 'adam' (default): additive on the box "
                             "[0,1]^E -- measured to NEVER reroute (Jaccard 1.0 support "
                             "vs init) because on a box every congestion loss is minimised "
                             "by shrinking x. 'eg': exponentiated-gradient / mirror descent "
                             "with a per-net mass constraint -- each net's total x is "
                             "pinned, so congestion can only fall by MOVING mass onto "
                             "connected alternatives (a simplex, not a box). Requires an "
                             "init with off-path x>0, e.g. --init-mode shortest_path "
                             "--init-off-path 0.01.")
    parser.add_argument("--eg-lr", type=float, default=0.5,
                        help="EG step size (dimensionless; gradient is per-net RMS-"
                             "normalised, so ~0.1-1 regardless of loss scale).")
    parser.add_argument("--eg-clip", type=float, default=1.0,
                        help="EG per-step exponent clip: one step moves any edge by at "
                             "most a factor exp(eg_clip).")
    parser.add_argument("--lam-update", choices=["meng", "mult"], default="meng",
                        help="Multiplier update. 'meng' (default): normalised subgradient "
                             "-- measured decorative (lam_max=0.0104, 190000x below the rho "
                             "term). 'mult': PathFinder-style history "
                             "lam<-lam*(1+eta*relu(u/c-1)), so persistently over-capacity "
                             "edges keep getting more expensive (a ratchet).")
    parser.add_argument("--lam-mult-eta", type=float, default=0.5,
                        help="--lam-update mult: geometric growth rate per outer.")
    parser.add_argument("--lam-base", type=float, default=1.0,
                        help="--lam-update mult: initial lam (0*anything=0, so mult needs "
                             "a positive seed).")
    parser.add_argument("--congestion-mode", choices=["hard", "soft"], default="hard",
                        help="Overflow signal. 'hard' (default): relu(usage-capacity) -- "
                             "ZERO gradient below capacity, so no preventive spreading "
                             "(measured: overflow exactly 0 at uniform init => lam/rho "
                             "never engage). 'soft': capacity-relative softplus, gradient "
                             "everywhere and scale-free across heterogeneous capacities "
                             "(mean 31.6, median 30); -> hard as tau->0.")
    parser.add_argument("--congestion-tau", type=float, default=0.1,
                        help="--congestion-mode soft: softness (fraction of capacity).")
    parser.add_argument("--flow-demand", choices=["fanout", "normalized"],
                        default="fanout",
                        help="Kirchhoff demand. 'fanout' (default): {src:-K, sink:+1} -- "
                             "measured 99.93%% x-independent constant, unsatisfiable for "
                             "5.2%% of nets (source must push K units through <=13 edges "
                             "with x<=1), and its ~-2K gradient on source edges generates "
                             "congestion at high-fanout sources. 'normalized': scaled by "
                             "1/K -> {src:-1, sink:+1/K}; satisfiable, and consistent "
                             "with x as an edge indicator (trunk edge carries 1).")
    parser.add_argument("--conn-sat-alpha", type=float, default=0.0,
                        help="Saturating connectivity (0 = off, plain sum of effective "
                             "resistance). If >0, the loss becomes "
                             "sum_col relu(R_eff - alpha*d_min), where d_min is the net's "
                             "min-hop distance (= the R_eff of one clean shortest path). "
                             "Plain sum(R_eff) keeps rewarding extra parallel paths "
                             "forever (parallel resistors); saturating stops the reward "
                             "once a net is connected well enough. Try 1.5-3.")
    parser.add_argument("--grad-balance", choices=["off", "conn"], default="off",
                        help="Rescale w_conn each outer so ||w_conn*grad_conn|| = "
                             "--balance-ratio * ||grad_wl||. Measured on boom_soc_v2: "
                             "wirelength 7.2e4, congestion penalty 2.7e7, effective "
                             "resistance 1.3e13-3.2e14 -- so ER is ~1e10x the rest and the "
                             "AL optimises it alone. lambda cannot fix that (it weights "
                             "the constraint, not the objective, and its normalised update "
                             "grows ||lambda|| by exactly lr_lam per outer).")
    parser.add_argument("--balance-ratio", type=float, default=1.0,
                        help="--grad-balance target: ||w_conn*grad_conn|| / ||grad_wl||.")
    parser.add_argument("--balance-every", type=int, default=1,
                        help="--grad-balance: rebalance every N outers (one extra "
                             "connectivity solve each time).")
    parser.add_argument("--conn-every", type=int, default=1,
                        help="A1: evaluate connectivity every K inner iters (1=every step)")
    parser.add_argument("--conn-freeze-outer", type=int, default=0,
                        help="A2: stop evaluating connectivity after this outer iter (0=never)")
    parser.add_argument("--early-stop-tol", type=float, default=0.0,
                        help="Stop when overflow improves less than this fraction over "
                             "--early-stop-patience outer iters (0=disabled, 0.01=1%%). "
                             "Adapts iteration count to the design instead of a fixed cap.")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="Outer iters the early-stop improvement is measured over")
    parser.add_argument("--overflow-stop-frac", type=float, default=0.0,
                        help="EG early stop: halt once relaxed overflow falls below "
                             "this fraction of its first-outer value (0=off). The EG "
                             "reroute converges when overflow crosses ~1%% of its "
                             "initial value; iterations past that only sharpen "
                             "magnitudes the guide ignores. Try 0.01 (totwl-driven).")
    parser.add_argument("--conn-super-sink", action="store_true",
                        help="B1: 1 connectivity column/net (source->merged super-sink)")
    parser.add_argument("--conn-bf16", action="store_true",
                        help="B4: run the connectivity CG in bfloat16")
    parser.add_argument("--conn-multi-gpu", type=int, default=1,
                        help="B2: shard connectivity groups across N GPUs (cuda:0..N-1)")
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
