#!/usr/bin/env python3
"""Sweep connectivity runtime levers on one loaded router (pay load cost once).

Measures the net-grouped connectivity forward+backward under different
cg_max_iter / col_chunk, and quantifies the 'drop low-fanout nets' lever by
showing how many RHS columns come from each fanout bucket.
"""
import argparse, os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--testcase", default="boom_soc_v2")
    p.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    p.add_argument("--edge-mode", default="directed")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--probe-groups", type=int, default=4000,
                   help="Groups to time per config (sampled from the front).")
    args = p.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path
    from src.router.connectivity_grouped import effective_resistance_loss_grouped

    dev = torch.device(f"cuda:{args.gpu}")
    ni = default_net_index_path("./data/", args.testcase, args.rrg, args.edge_mode,
                                0, 0.1, route_filter="stubs", edge_scope="corridor",
                                corridor_width=2, max_edges_per_net=50000)
    t0 = time.time()
    router = GlobalRouter.load(args.rrg, ni, device=dev, edge_mode=args.edge_mode, verbose=True)
    print(f"[load] {time.time()-t0:.1f}s  nets={router.num_nets} vars={router.num_vars}", flush=True)

    x = router.init_variables()
    conn = router._conn
    vo = torch.tensor(router._var_offset, dtype=torch.long, device=dev)
    no = torch.tensor(router._node_offset, dtype=torch.long, device=dev)

    # ---- column distribution by per-net fanout (columns) ----
    col_net = torch.searchsorted(no, conn["src_flat"].long(), right=True) - 1
    net_ncol = torch.bincount(col_net, minlength=router.num_nets).cpu().numpy()
    total_cols = int(net_ncol.sum())
    print(f"\n[cols] total RHS columns (num_cols) = {total_cols}", flush=True)
    buckets = [(1,1),(2,2),(3,4),(5,8),(9,16),(17,10**9)]
    for lo, hi in buckets:
        m = (net_ncol >= lo) & (net_ncol <= hi)
        nnets = int(m.sum()); ncols = int(net_ncol[m].sum())
        lbl = f"{lo}" if lo == hi else (f"{lo}+" if hi > 10**8 else f"{lo}-{hi}")
        print(f"  fanout(cols)={lbl:>5}: {nnets:>7} nets ({100*nnets/router.num_nets:4.1f}%)  "
              f"{ncols:>9} cols ({100*ncols/total_cols:4.1f}% of columns)", flush=True)
    # cumulative: columns removed if we drop nets with <= k columns
    print("  --- if connectivity SKIPS nets with fanout(cols) <= k ---", flush=True)
    for k in (1, 2, 4):
        keep_cols = int(net_ncol[net_ncol > k].sum())
        print(f"    drop <= {k}: keep {keep_cols} cols "
              f"({100*keep_cols/total_cols:.1f}%)  -> ~{100*keep_cols/total_cols:.1f}% of runtime", flush=True)

    def bench(cg_iter, col_chunk, ng_probe, cg_tol=1e-4):
        gcache = {}
        # warm-up: build group table + one group
        _ = effective_resistance_loss_grouped(x.detach(), conn, vo, no, eps=1e-6,
                cg_max_iter=cg_iter, cg_tol=cg_tol, col_chunk=col_chunk,
                max_groups=1, _group_cache=gcache)
        ng = gcache["num_groups"]
        torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
        xg = x.detach().clone().requires_grad_(True)
        t = time.time()
        L = effective_resistance_loss_grouped(xg, conn, vo, no, eps=1e-6,
                cg_max_iter=cg_iter, cg_tol=cg_tol, col_chunk=col_chunk,
                max_groups=ng_probe, _group_cache=gcache)
        L.backward(); torch.cuda.synchronize(dev)
        dt = time.time() - t
        peak = torch.cuda.max_memory_allocated(dev) / 1e6
        per = dt / ng_probe
        full = per * ng
        print(f"[bench] cg_iter={cg_iter:3d} col_chunk={col_chunk:3d}  groups={ng:6d}  "
              f"per-group={per*1000:6.2f}ms  peak={peak:7.0f}MB  "
              f"=> FULL fwd+bwd ~{full:6.1f}s ({full/60:.1f} min)", flush=True)
        return full

    print("\n[sweep] full = extrapolated to all groups (fwd+bwd, one connectivity eval)", flush=True)
    for cc in (32, 128, 512):
        for it in (100, 50, 20, 10):
            bench(it, cc, args.probe_groups)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
