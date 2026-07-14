#!/usr/bin/env python3
"""Measure how connectivity runtime responds to fanout-filter / sink-cap / cg_iter
/ col_chunk levers, to see how close to a ~3-min run we can get.

Filters the connectivity RHS columns (source->sink pairs) and times grouped CG
fwd+bwd on a sample of groups, extrapolated to the full group count.
"""
import argparse, os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    ap.add_argument("--probe-groups", type=int, default=3000)
    args = ap.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path
    from src.router.connectivity_grouped import effective_resistance_loss_grouped

    dev = torch.device(f"cuda:{args.gpu}")
    ni = default_net_index_path("./data/", "boom_soc_v2", args.rrg, "directed", 0, 0.1,
                                route_filter="stubs", edge_scope="corridor",
                                corridor_width=2, max_edges_per_net=50000)
    t0 = time.time()
    router = GlobalRouter.load(args.rrg, ni, device=dev, edge_mode="directed", verbose=True)
    print(f"[load] {time.time()-t0:.1f}s nets={router.num_nets}", flush=True)

    x = router.init_variables().detach()
    conn0 = router._conn
    vo = torch.tensor(router._var_offset, dtype=torch.long, device=dev)
    no = torch.tensor(router._node_offset, dtype=torch.long, device=dev)
    src = conn0["src_flat"].long()
    col_net = torch.searchsorted(no, src, right=True) - 1
    net_ncol = torch.bincount(col_net, minlength=router.num_nets)
    net_c_off = torch.zeros(router.num_nets + 1, dtype=torch.long, device=dev)
    net_c_off[1:] = torch.cumsum(net_ncol, 0)
    rank_in_net = torch.arange(src.numel(), device=dev) - net_c_off[col_net]
    total_cols0 = src.numel()
    print(f"[cols] baseline num_cols={total_cols0}", flush=True)

    def make_conn(mask):
        m = mask.nonzero(as_tuple=True)[0]
        n = m.numel()
        return dict(flat_u=conn0["flat_u"], flat_v=conn0["flat_v"],
                    src_flat=conn0["src_flat"][m], sink_flat=conn0["sink_flat"][m],
                    col_id=torch.arange(n, device=dev), num_nodes=conn0["num_nodes"],
                    num_cols=n), n

    def bench(name, conn, n, cg_iter, col_chunk):
        gc = {}
        _ = effective_resistance_loss_grouped(x, conn, vo, no, eps=1e-6, cg_max_iter=cg_iter,
                cg_tol=1e-4, col_chunk=col_chunk, max_groups=1, _group_cache=gc)
        ng = gc["num_groups"]
        torch.cuda.synchronize(dev)
        xg = x.clone().requires_grad_(True)
        t = time.time()
        L = effective_resistance_loss_grouped(xg, conn, vo, no, eps=1e-6, cg_max_iter=cg_iter,
                cg_tol=1e-4, col_chunk=col_chunk, max_groups=args.probe_groups, _group_cache=gc)
        L.backward(); torch.cuda.synchronize(dev)
        per = (time.time() - t) / args.probe_groups
        full = per * ng
        print(f"  {name:32s} cols={n:8d} groups={ng:6d} cg_iter={cg_iter} cc={col_chunk} "
              f"=> full fwd+bwd ~{full:6.1f}s ({full/60:.1f}min)", flush=True)

    all_mask = torch.ones(total_cols0, dtype=torch.bool, device=dev)
    print("\n[levers] full connectivity fwd+bwd, extrapolated:", flush=True)
    # baseline
    c, n = make_conn(all_mask); bench("baseline (all nets, cg8/cc128)", c, n, 8, 128)
    # fanout filters
    for K in (5, 9, 17):
        c, n = make_conn(net_ncol[col_net] >= K); bench(f"fanout>={K} (cg8/cc128)", c, n, 8, 128)
    # sink caps
    for M in (16, 8, 4):
        c, n = make_conn(rank_in_net < M); bench(f"cap {M} sinks/net (cg8/cc128)", c, n, 8, 128)
    # stacked fast config: cap8 + cg5 + cc512
    c, n = make_conn(rank_in_net < 8); bench("STACK cap8 + cg5 + cc512", c, n, 5, 512)
    c, n = make_conn((rank_in_net < 8) & (net_ncol[col_net] >= 2))
    bench("STACK cap8 + drop fanout1 + cg5/cc512", c, n, 5, 512)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
