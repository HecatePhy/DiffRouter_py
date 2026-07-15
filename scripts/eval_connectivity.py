#!/usr/bin/env python3
"""Fast routing-quality metric: fraction of nets whose sinks are all reachable
from the source in the thresholded (x >= thr) graph.

Sidesteps the slow Steiner-ordering extractor. Uses GPU label-propagation
connected components over all nets at once (per-net node blocks are disjoint in
the connectivity local-node space), then checks src<->sink same-component.
"""
import argparse, os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch


def connected_fraction(router, x, thresh, max_iter=2000):
    """Return (frac_nets_fully_connected, frac_sink_pairs_connected)."""
    conn = router._conn
    if conn is None:
        return float("nan"), float("nan")
    dev = x.device
    fu = conn["flat_u"].long(); fv = conn["flat_v"].long()
    num_nodes = conn["num_nodes"]
    # active edges (x aligns 1:1 with flat_u/flat_v in variable order)
    mask = x >= thresh
    au = fu[mask]; av = fv[mask]
    label = torch.arange(num_nodes, device=dev, dtype=torch.long)
    for _ in range(max_iter):
        new = label.clone()
        new.scatter_reduce_(0, au, label[av], reduce="amin")
        new.scatter_reduce_(0, av, label[au], reduce="amin")
        if torch.equal(new, label):
            break
        label = new
    src = conn["src_flat"].long(); sink = conn["sink_flat"].long()
    col_ok = label[src] == label[sink]                       # [num_cols]
    # map each column to its net via node_offset (sorted) on the src node id
    node_off = torch.tensor(router._node_offset, dtype=torch.long, device=dev)
    col_net = torch.searchsorted(node_off, src, right=True) - 1
    num_nets = len(router._node_offset) - 1
    ncol = torch.bincount(col_net, minlength=num_nets).clamp_min(1)
    ok_per_net = torch.zeros(num_nets, device=dev, dtype=torch.long)
    ok_per_net.index_add_(0, col_net, col_ok.long())
    net_full = (ok_per_net == ncol) & (ncol > 0)
    have_cols = torch.bincount(col_net, minlength=num_nets) > 0
    frac_nets = net_full[have_cols].float().mean().item()
    frac_pairs = col_ok.float().mean().item()
    return frac_nets, frac_pairs, int(have_cols.sum()), int(col_ok.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    ap.add_argument("--testcase", default="boom_soc_v2")
    ap.add_argument("--x", nargs="+", required=True,
                    help="label=path/to/global_x.pt pairs")
    args = ap.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path
    dev = torch.device(f"cuda:{args.gpu}")
    ni = default_net_index_path("./data/", args.testcase, args.rrg, "directed", 0, 0.1,
                                route_filter="stubs", edge_scope="corridor",
                                corridor_width=2, max_edges_per_net=50000)
    t0 = time.time()
    router = GlobalRouter.load(args.rrg, ni, device=dev, edge_mode="directed", verbose=True)
    print(f"[load] {time.time()-t0:.1f}s  nets={router.num_nets}", flush=True)

    for spec in args.x:
        label, path = spec.split("=", 1)
        obj = torch.load(path, map_location=dev)
        x = (obj["x"] if isinstance(obj, dict) else obj).to(dev)
        print(f"\n=== {label}  ({path}) ===", flush=True)
        frac_disc = (x * (1 - x)).sum().item() / x.numel()   # avg x(1-x): 0 = fully discrete
        print(f"  x stats: min={x.min():.3f} max={x.max():.3f} mean={x.mean():.4f} "
              f">0.01: {(x>=0.01).float().mean()*100:.1f}%  >0.1: {(x>=0.1).float().mean()*100:.1f}%  "
              f">0.5: {(x>=0.5).float().mean()*100:.2f}%  frac_x(1-x)={frac_disc:.4f}", flush=True)
        # congestion/overflow (routability): the other half of quality
        with torch.no_grad():
            _, ovf = router._get_usage_and_overflows(x)
            n_over = int((ovf > 1e-6).sum())
            print(f"  overflow: max={ovf.max().item():.4f}  sum={ovf.sum().item():.1f}  "
                  f"edges_over={n_over} ({100*n_over/ovf.numel():.2f}% of phys edges)", flush=True)
        for thr in (0.01, 0.05, 0.1, 0.3):
            t = time.time()
            fn, fp, nnets, npairs = connected_fraction(router, x, thr)
            print(f"  thr={thr:4.2f}: nets_fully_connected={fn*100:6.2f}%  "
                  f"sink_pairs_connected={fp*100:6.2f}%  "
                  f"(nets={nnets}, pairs={npairs})  {time.time()-t:.1f}s", flush=True)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
