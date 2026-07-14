#!/usr/bin/env python3
"""Profile GPU memory + variable distribution for one global-routing AL iteration.

Loads RRG + net index like run_exp.py, moves variables to GPU, and reports
torch.cuda peak memory after each loss term, the backward pass, and a simulated
Adam step -- plus the per-net variable-mass distribution that decides which
memory-reduction ideas pay off.
"""

import argparse
import os
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.chdir(_root)

import numpy as np
import torch


def mb(nbytes: int) -> float:
    return nbytes / 1e6


def cuda_peak_mb(dev) -> float:
    return torch.cuda.max_memory_allocated(dev) / 1e6


def cuda_cur_mb(dev) -> float:
    return torch.cuda.memory_allocated(dev) / 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--testcase", default="boom_soc_v2")
    p.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    p.add_argument("--net-index", default=None)
    p.add_argument("--edge-mode", default="directed")
    p.add_argument("--edge-scope", choices=["bbox", "corridor"], default="corridor")
    p.add_argument("--corridor-width", type=int, default=2)
    p.add_argument("--max-edges-per-net", type=int, default=50000)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--conn-col-chunk", type=int, default=32)
    p.add_argument("--conn-cg-max-iter", type=int, default=100)
    p.add_argument("--conn-edge-chunk", type=int, default=0,
                   help="Bound Laplacian matvec temporary to [edge_chunk, col_chunk]")
    p.add_argument("--conn-probe-chunks", type=int, default=3,
                   help="Only run this many column-chunks of connectivity (0=all). "
                        "Peak memory is per-chunk, so a few chunks give the true peak fast.")
    p.add_argument("--grouped-probe-groups", type=int, default=0,
                   help="Also benchmark the net-grouped prototype on this many groups "
                        "(0=skip). Extrapolates full grouped forward+backward time.")
    args = p.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path

    net_index = args.net_index or default_net_index_path(
        "./data/", args.testcase, args.rrg, args.edge_mode, 0, 0.1,
        route_filter="stubs",
        edge_scope=args.edge_scope,
        corridor_width=args.corridor_width,
        max_edges_per_net=args.max_edges_per_net,
    )

    dev = torch.device(f"cuda:{args.gpu}")
    print(f"[gpu-profile] testcase={args.testcase} edge_mode={args.edge_mode} "
          f"scope={args.edge_scope} corr={args.corridor_width} dev={dev}", flush=True)

    # --- Load router on CPU first (net index is huge), then move loss tensors to GPU
    t0 = time.time()
    router = GlobalRouter.load(args.rrg, net_index, device=dev,
                               edge_mode=args.edge_mode, verbose=True)
    router.conn_col_chunk = args.conn_col_chunk
    router.conn_cg_max_iter = args.conn_cg_max_iter
    router.conn_edge_chunk = args.conn_edge_chunk
    print(f"[gpu-profile] build_router: {time.time()-t0:.1f}s  "
          f"edge_chunk={args.conn_edge_chunk}", flush=True)
    print(f"[gpu-profile] nets={router.num_nets} vars={router.num_vars}", flush=True)

    # ---------- variable-mass distribution ----------
    offs = np.asarray(router._var_offset, dtype=np.int64)
    sizes = np.diff(offs)          # vars per net
    sizes_sorted = np.sort(sizes)[::-1]
    total = int(sizes.sum())
    n = len(sizes)
    cum = np.cumsum(sizes_sorted)
    def frac_nets_for(mass_frac):
        target = mass_frac * total
        k = int(np.searchsorted(cum, target)) + 1
        return k, 100.0 * k / n
    print("\n[dist] per-net variable counts:", flush=True)
    print(f"  nets={n}  total_vars={total}  mean={total/n:.1f}  "
          f"median={np.median(sizes):.0f}  max={sizes.max()}  min={sizes.min()}", flush=True)
    for q in (50, 90, 99, 100):
        print(f"  p{q}={np.percentile(sizes, q):.0f}", flush=True)
    for mf in (0.5, 0.8, 0.9):
        k, pct = frac_nets_for(mf)
        print(f"  top {k} nets ({pct:.2f}% of nets) hold {int(mf*100)}% of variables", flush=True)
    # how many vars live in nets below various sizes (prune candidates)
    for thr in (10, 50, 100, 500):
        small = sizes[sizes <= thr]
        print(f"  nets with <= {thr} vars: {len(small)} nets "
              f"({100*len(small)/n:.1f}%), holding {int(small.sum())} vars "
              f"({100*small.sum()/total:.1f}% of total)", flush=True)

    # ---------- persistent tensor footprint ----------
    print("\n[persistent] flat tensors on GPU:", flush=True)
    persistent = 0
    for name in ("_flat_edge_idx", "_flat_wl", "_flat_u", "_flat_v", "_flat_wl"):
        t = getattr(router, name, None)
        if torch.is_tensor(t):
            persistent += t.element_size() * t.nelement()
    for k in ("flat_u", "flat_v", "src_flat", "sink_flat", "col_id"):
        if router._conn and torch.is_tensor(router._conn.get(k)):
            t = router._conn[k]
    print(f"  flat loss tensors ~ {mb(persistent):.1f} MB", flush=True)

    torch.cuda.reset_peak_memory_stats(dev)
    x = router.init_variables()
    base = cuda_cur_mb(dev)
    print(f"[step] x created: cur={base:.1f}MB  (x = {mb(x.element_size()*x.nelement()):.1f}MB)", flush=True)

    def stage(label, fn, keep_graph=False):
        torch.cuda.reset_peak_memory_stats(dev)
        t = time.time()
        out = fn()
        torch.cuda.synchronize(dev)
        peak = cuda_peak_mb(dev)
        val = float(out.item()) if torch.is_tensor(out) and out.numel() == 1 else None
        print(f"[step] {label}: {time.time()-t:.2f}s  peak={peak:.1f}MB  cur={cuda_cur_mb(dev):.1f}MB"
              + (f"  value={val:.4f}" if val is not None else ""), flush=True)
        return out

    wl = stage("wirelength", lambda: router.wirelength_loss(x))
    ovf = stage("usage+overflow", lambda: router._get_usage_and_overflows(x)[1].sum())

    # --- connectivity: measure peak + per-col-chunk time on a few chunks, extrapolate
    import src.router.connectivity as C
    ncols = router._conn["num_cols"] if router._conn else 0
    nchunks = (ncols + args.conn_col_chunk - 1) // max(1, args.conn_col_chunk)
    print(f"[conn] num_cols={ncols}  col_chunk={args.conn_col_chunk}  "
          f"=> {nchunks} sequential CG solves  (cg_max_iter={args.conn_cg_max_iter})", flush=True)
    C.PROFILE_MAX_COL_CHUNKS = args.conn_probe_chunks
    tconn = time.time()
    conn = stage(f"connectivity(cg, {args.conn_probe_chunks} of {nchunks} chunks)",
                 lambda: router.connectivity_loss_effective_resistance(x, solver="cg"))
    conn_probe_t = time.time() - tconn
    probed = max(1, args.conn_probe_chunks)
    per_chunk = conn_probe_t / probed
    print(f"[conn] per-chunk ~{per_chunk:.2f}s  => full connectivity forward "
          f"~{per_chunk*nchunks/60:.1f} min for all {nchunks} chunks", flush=True)
    C.PROFILE_MAX_COL_CHUNKS = 0

    # --- grouped prototype: run a few groups, extrapolate full forward time
    if args.grouped_probe_groups > 0:
        from src.router.connectivity_grouped import effective_resistance_loss_grouped
        vo = torch.tensor(router._var_offset, dtype=torch.long, device=dev)
        no = torch.tensor(router._node_offset, dtype=torch.long, device=dev)
        gcache = {}
        # warm-up group build (builds group table, processes 0 groups of work)
        _ = effective_resistance_loss_grouped(
            x.detach(), router._conn, vo, no, eps=1e-6,
            cg_max_iter=args.conn_cg_max_iter, cg_tol=1e-5,
            col_chunk=args.conn_col_chunk, max_groups=1, _group_cache=gcache)
        ngroups = gcache.get("num_groups", 0)
        print(f"[grouped] num_groups={ngroups} (col_chunk={args.conn_col_chunk})", flush=True)
        torch.cuda.reset_peak_memory_stats(dev)
        xg = x.detach().clone().requires_grad_(True)
        tg = time.time()
        Lg = effective_resistance_loss_grouped(
            xg, router._conn, vo, no, eps=1e-6,
            cg_max_iter=args.conn_cg_max_iter, cg_tol=1e-5,
            col_chunk=args.conn_col_chunk,
            max_groups=args.grouped_probe_groups, _group_cache=gcache)
        Lg.backward()
        torch.cuda.synchronize(dev)
        tg = time.time() - tg
        gpeak = cuda_peak_mb(dev)
        per_g = tg / args.grouped_probe_groups
        print(f"[grouped] {args.grouped_probe_groups} groups fwd+bwd: {tg:.2f}s  "
              f"peak={gpeak:.1f}MB  per-group~{per_g*1000:.1f}ms  value={Lg.item():.4f}", flush=True)
        print(f"[grouped] => full grouped forward+backward "
              f"~{per_g*ngroups/60:.2f} min for all {ngroups} groups", flush=True)

    flow = stage("flow_conservation", lambda: router.flow_conservation_loss(x))

    # full AL objective + backward (connectivity capped to probe chunks so this
    # completes in seconds; peak memory is per-chunk so it is still the true peak)
    C.PROFILE_MAX_COL_CHUNKS = args.conn_probe_chunks
    torch.cuda.reset_peak_memory_stats(dev)
    lam = torch.zeros(len(router.rrg.phys_list), device=dev, dtype=x.dtype)
    t = time.time()
    L = router.augmented_lagrangian(x, lam, rho=1.0)
    fwd_peak = cuda_peak_mb(dev)
    print(f"[step] AL forward: {time.time()-t:.2f}s  peak={fwd_peak:.1f}MB", flush=True)
    torch.cuda.reset_peak_memory_stats(dev)
    t = time.time()
    L.backward()
    bwd_peak = cuda_peak_mb(dev)
    print(f"[step] AL backward: {time.time()-t:.2f}s  peak={bwd_peak:.1f}MB  grad_norm={x.grad.norm().item():.4f}", flush=True)
    C.PROFILE_MAX_COL_CHUNKS = 0

    # simulate Adam state cost
    adam_state = 2 * x.element_size() * x.nelement()
    print(f"\n[optimizer] Adam extra state (2 buffers) ~ {mb(adam_state):.1f}MB "
          f"(SGD-momentum would be {mb(adam_state/2):.1f}MB, plain SGD 0)", flush=True)
    print(f"[summary] vars={router.num_vars}  x={mb(x.element_size()*x.nelement()):.1f}MB  "
          f"AL_fwd_peak={fwd_peak:.1f}MB  AL_bwd_peak={bwd_peak:.1f}MB", flush=True)
    print("[gpu-profile] done", flush=True)


if __name__ == "__main__":
    main()
