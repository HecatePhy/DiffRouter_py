#!/usr/bin/env python3
"""Validate connectivity quality vs CG iteration count.

Compares the net-grouped effective-resistance loss AND its gradient at reduced
cg_max_iter against a well-converged reference (many iters, tight tol), on a
real sample of boom_soc_v2 groups. Gradient direction (cosine) is what matters
for optimization, so we report that alongside relative errors.
"""
import argparse, os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--testcase", default="boom_soc_v2")
    p.add_argument("--rrg", default="data/rrg_xcvu3p_int.pt")
    p.add_argument("--edge-mode", default="directed")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--col-chunk", type=int, default=128)
    p.add_argument("--sample-groups", type=int, default=4000)
    p.add_argument("--ref-iters", type=int, default=300)
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
    print(f"[load] {time.time()-t0:.1f}s  nets={router.num_nets}", flush=True)

    conn = router._conn
    vo = torch.tensor(router._var_offset, dtype=torch.long, device=dev)
    no = torch.tensor(router._node_offset, dtype=torch.long, device=dev)
    gcache = {}

    # Use two initial-x settings: the uniform init, and a random one (mid-optimization-like).
    def make_x(kind):
        if kind == "init":
            return router.init_variables().detach()
        # pseudo-random but deterministic-ish (Math.random not needed): hash of arange
        idx = torch.arange(router.num_vars, device=dev, dtype=torch.float32)
        return (0.5 + 0.5 * torch.sin(idx * 0.7)).clamp_(0.02, 0.98)

    def loss_grad(x0, iters, tol):
        xg = x0.clone().requires_grad_(True)
        L = effective_resistance_loss_grouped(xg, conn, vo, no, eps=1e-6,
                cg_max_iter=iters, cg_tol=tol, col_chunk=args.col_chunk,
                max_groups=args.sample_groups, _group_cache=gcache)
        L.backward()
        return L.item(), xg.grad.detach()

    for kind in ("init", "random"):
        x0 = make_x(kind)
        print(f"\n=== x = {kind}  (sample={args.sample_groups} groups, "
              f"col_chunk={args.col_chunk}, ref={args.ref_iters} iters) ===", flush=True)
        t = time.time()
        Lref, gref = loss_grad(x0, args.ref_iters, 1e-12)
        # restrict to edges with nonzero reference grad (the probed groups)
        mask = gref != 0
        gref_m = gref[mask]
        gref_norm = gref_m.norm()
        print(f"  ref: loss={Lref:.6e}  |grad|={gref_norm.item():.4e}  "
              f"nnz_edges={int(mask.sum())}  ({time.time()-t:.1f}s)", flush=True)
        print(f"  {'iters':>5} {'relLossErr':>11} {'gradRelL2':>10} {'gradCosSim':>11} {'time':>7}", flush=True)
        for it in (100, 50, 30, 20, 10, 5):
            t = time.time()
            L, g = loss_grad(x0, it, 1e-12)
            gm = g[mask]
            rel_loss = abs(L - Lref) / (abs(Lref) + 1e-30)
            rel_g = (gm - gref_m).norm().item() / (gref_norm.item() + 1e-30)
            cos = torch.dot(gm, gref_m).item() / (gm.norm().item() * gref_norm.item() + 1e-30)
            print(f"  {it:5d} {rel_loss:11.3e} {rel_g:10.3e} {cos:11.6f} {time.time()-t:6.1f}s", flush=True)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
