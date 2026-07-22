#!/usr/bin/env python3
"""Does guide extraction destroy the optimiser's congestion information?

Compares three congestion pictures on the same design:

  1. relaxed-opt   : congestion of the optimised x itself (what the optimiser achieved)
  2. extracted-opt : congestion of the DISCRETE paths extracted from that x
  3. extracted-ctrl: congestion of the discrete paths from a uniform x (no optimisation)

Extraction takes each net's shortest path *independently*, so any congestion
coordination the optimiser found can be lost. If (1) is much better than (2), and
(2) is no better than (3), extraction is throwing the optimisation away.

Congestion is measured on the same physical edges the router constrains, so the
numbers are directly comparable to the AL's own overflow.

Writes a PNG with the three maps + the opt-vs-ctrl difference.
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


def path_edge_usage(router, x, dev, eps=1e-6):
    """Per-phys-edge usage of the extracted shortest paths (one path per net,sink).

    Reuses the guide's SSSP + predecessor, then marks the variables on the chosen
    paths, so this is exactly the routing the guide encodes.
    """
    from src.router.gpu_guide import compute_guide_paths  # noqa: F401  (same math)

    conn = router._conn
    fu = conn["flat_u"].long().to(dev)
    fv = conn["flat_v"].long().to(dev)
    n_nodes = int(conn["num_nodes"])
    src_col = conn["src_flat"].long().to(dev)
    sink_col = conn["sink_flat"].long().to(dev)

    xd = x.to(dev).clamp_min(0)
    cost = 1.0 / (xd + eps)

    dist = torch.full((n_nodes,), float("inf"), device=dev)
    dist[src_col] = 0.0
    for _ in range(400):
        new = dist.clone()
        new.scatter_reduce_(0, fv, dist[fu] + cost, reduce="amin")
        new.scatter_reduce_(0, fu, dist[fv] + cost, reduce="amin")
        if torch.equal(new, dist):
            break
        dist = new

    # Reconstruct the SAME predecessor tree the guide export uses, then count only the
    # edges on the paths actually walked from the sinks.
    #
    # NOTE: counting every "tight" edge (dist[u]+cost == dist[v], i.e. the whole
    # shortest-path DAG) is WRONG here: with uniform x every edge ties, so the DAG
    # explodes and the uniform baseline looks absurdly congested. pred is a single-parent
    # tree, so each visited node contributes exactly one edge -- that is the real route.
    tol = 1e-6
    big = torch.full((n_nodes,), n_nodes, dtype=torch.long, device=dev)
    ok_v = (dist[fu] + cost - dist[fv]).abs() <= tol * dist[fv].clamp_min(1.0)
    ok_u = (dist[fv] + cost - dist[fu]).abs() <= tol * dist[fu].clamp_min(1.0)
    big.scatter_reduce_(0, fv[ok_v], fu[ok_v], reduce="amin")
    big.scatter_reduce_(0, fu[ok_u], fv[ok_u], reduce="amin")
    pred = torch.where(big < n_nodes, big, torch.full_like(big, -1))
    pred[src_col] = -1

    visited = torch.zeros(n_nodes, dtype=torch.bool, device=dev)
    visited[sink_col] = True
    cur = sink_col.clone()
    alive = torch.ones_like(cur, dtype=torch.bool)
    for _ in range(4000):
        nxt = torch.where(alive, pred[cur], torch.full_like(cur, -1))
        alive = alive & (nxt >= 0)
        if not bool(alive.any()):
            break
        cur = torch.where(alive, nxt, cur)
        visited[cur[alive]] = True

    # edge e is used iff it is the tree edge into a visited node, counting ONLY the
    # traversal-aligned orientation (pred[v] -> v). Directed pairs both exist in the
    # corridor, and the old two-clause test marked both -- double-counting every path
    # edge against physical capacity (same bug as gpu_guide.shortest_path_x).
    used = ((pred[fv] == fu) & visited[fv]).float()

    # scatter to physical edges exactly as the router's own overflow does
    num_phys = len(router.rrg.phys_list)
    usage_d = torch.zeros(router._num_global_edges, device=dev)
    usage_d.scatter_add_(0, router._flat_edge_idx.long().to(dev), used)
    phys = torch.zeros(num_phys, device=dev)
    phys.index_add_(0, router._d2p.long().to(dev), usage_d)
    return phys


def stats(name, phys_usage, cap):
    ovf = torch.relu(phys_usage - cap)
    over = int((ovf > 1e-6).sum())
    print(f"  {name:16s} overflow_sum={float(ovf.sum()):12.0f}  max={float(ovf.max()):7.1f}  "
          f"edges_over={over:7d} ({100*over/ovf.numel():5.2f}%)", flush=True)
    return ovf


def to_grid(router, per_edge, R, C):
    ends = np.asarray(router.rrg.phys_list, dtype=np.int64).reshape(-1, 2)
    v = per_edge.detach().cpu().numpy()
    tile = np.zeros(router.rrg.num_tiles, dtype=np.float32)
    np.maximum.at(tile, ends[:, 0], v)
    np.maximum.at(tile, ends[:, 1], v)
    g = np.zeros((R, C), dtype=np.float32)
    for idx, t in enumerate(router.rrg.tiles):
        r, c = t[0], t[1]
        if 0 <= r < R and 0 <= c < C:
            g[r, c] = tile[idx]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testcase", default="boom_soc_v2")
    ap.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    ap.add_argument("--x", required=True, help="optimised global_x.pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="congestion_diag.png")
    args = ap.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path

    dev = torch.device(f"cuda:{args.gpu}")
    ni = default_net_index_path("./data/", args.testcase, args.rrg, "directed", 0, 0.1,
                                route_filter="stubs", edge_scope="corridor",
                                corridor_width=2, max_edges_per_net=50000)
    t = time.time()
    router = GlobalRouter.load(args.rrg, ni, device=dev, edge_mode="directed", verbose=False)
    print(f"[load] {time.time()-t:.1f}s  nets={router.num_nets}", flush=True)
    cap = router._phys_capacity_tensor.to(dev)

    obj = torch.load(args.x, map_location=dev)
    x_opt = (obj["x"] if isinstance(obj, dict) else obj).to(dev)
    x_ctrl = router.init_variables().detach()

    print("\n=== congestion on physical edges (same measure the AL constrains) ===", flush=True)
    # 1. the relaxed solution itself
    with torch.no_grad():
        _, ovf_relaxed = router._get_usage_and_overflows(x_opt)
    print(f"  {'relaxed-opt':16s} overflow_sum={float(ovf_relaxed.sum()):12.0f}  "
          f"max={float(ovf_relaxed.max()):7.1f}  "
          f"edges_over={int((ovf_relaxed>1e-6).sum()):7d}   <- what the optimiser achieved",
          flush=True)

    # 2/3. the discrete extracted paths
    u_opt = path_edge_usage(router, x_opt, dev)
    o_opt = stats("extracted-opt", u_opt, cap)
    u_ctrl = path_edge_usage(router, x_ctrl, dev)
    o_ctrl = stats("extracted-ctrl", u_ctrl, cap)

    d = float(o_opt.sum()) - float(o_ctrl.sum())
    base = max(float(o_ctrl.sum()), 1e-9)
    print(f"\n  extracted-opt vs extracted-ctrl: {d:+.0f} ({100*d/base:+.2f}%)", flush=True)
    print("  -> if ~0, extraction has discarded the optimiser's congestion work", flush=True)

    # ---- visualise
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        R, C = router.device_rows, router.device_cols
        g_rel = to_grid(router, ovf_relaxed, R, C)
        g_opt = to_grid(router, o_opt, R, C)
        g_ctl = to_grid(router, o_ctrl, R, C)
        g_dif = g_opt - g_ctl
        fig, ax = plt.subplots(1, 4, figsize=(22, 5))
        for a, g, t_ in [(ax[0], g_rel, "relaxed-opt overflow\n(optimiser's own solution)"),
                         (ax[1], g_ctl, "extracted-ctrl overflow\n(uniform x, no optimisation)"),
                         (ax[2], g_opt, "extracted-opt overflow\n(paths from optimised x)")]:
            im = a.imshow(g, cmap="hot", interpolation="nearest")
            a.set_title(t_, fontsize=10); plt.colorbar(im, ax=a, fraction=0.03)
        m = np.abs(g_dif).max() or 1.0
        im = ax[3].imshow(g_dif, cmap="bwr", vmin=-m, vmax=m, interpolation="nearest")
        ax[3].set_title("extracted-opt MINUS extracted-ctrl\n(blue = optimisation helped)", fontsize=10)
        plt.colorbar(im, ax=ax[3], fraction=0.03)
        plt.tight_layout()
        plt.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"\n[viz] {args.out}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] skipped: {exc}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
