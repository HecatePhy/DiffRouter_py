#!/usr/bin/env python3
"""Does Jacobi preconditioning fix CG convergence for a sharpened (hard) x?

On a real boom_soc_v2 group sample, compare unpreconditioned vs Jacobi CG at
low iteration counts against a well-converged reference (Jacobi, many iters).
Reports gradient cosine similarity + relative errors -- the metric that governs
whether the optimizer steps the right direction.
"""
import argparse, os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--col-chunk", type=int, default=128)
    p.add_argument("--sample-groups", type=int, default=4000)
    p.add_argument("--ref-iters", type=int, default=800)
    args = p.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path
    from src.router.connectivity_grouped import effective_resistance_loss_grouped

    dev = torch.device(f"cuda:{args.gpu}")
    ni = default_net_index_path("./data/", "boom_soc_v2", "data/rrg_xcvu3p_int.pt",
                                "directed", 0, 0.1, route_filter="stubs",
                                edge_scope="corridor", corridor_width=2,
                                max_edges_per_net=50000)
    t0 = time.time()
    router = GlobalRouter.load("data/rrg_xcvu3p_int.pt", ni, device=dev,
                               edge_mode="directed", verbose=True)
    print(f"[load] {time.time()-t0:.1f}s  nets={router.num_nets}", flush=True)

    conn = router._conn
    vo = torch.tensor(router._var_offset, dtype=torch.long, device=dev)
    no = torch.tensor(router._node_offset, dtype=torch.long, device=dev)
    gcache = {}

    # HARD x: wide weight spread (a sharpened, near-0/near-1 mid-optimization state)
    idx = torch.arange(router.num_vars, device=dev, dtype=torch.float32)
    x0 = (0.5 + 0.49 * torch.sin(idx * 0.7)).clamp_(0.02, 0.98)

    def loss_grad(iters, tol, precond):
        xg = x0.clone().requires_grad_(True)
        L = effective_resistance_loss_grouped(xg, conn, vo, no, eps=1e-6,
                cg_max_iter=iters, cg_tol=tol, col_chunk=args.col_chunk,
                max_groups=args.sample_groups, precond=precond, _group_cache=gcache)
        L.backward()
        return L.item(), xg.grad.detach()

    print(f"\n=== HARD x, sample={args.sample_groups} groups, col_chunk={args.col_chunk} ===", flush=True)
    t = time.time()
    Lref, gref = loss_grad(args.ref_iters, 1e-13, "jacobi")
    mask = gref != 0
    gref_m = gref[mask]; gn = gref_m.norm()
    print(f"  reference: jacobi {args.ref_iters} iters  loss={Lref:.6e}  "
          f"|grad|={gn.item():.4e}  ({time.time()-t:.1f}s)", flush=True)
    print(f"  {'precond':>8} {'iters':>5} {'relLossErr':>11} {'gradRelL2':>10} {'gradCosSim':>11} {'time':>7}", flush=True)
    for precond in ("none", "jacobi"):
        for it in (100, 50, 20, 10, 5):
            t = time.time()
            L, g = loss_grad(it, 1e-13, precond)
            gm = g[mask]
            rl = abs(L - Lref) / (abs(Lref) + 1e-30)
            rg = (gm - gref_m).norm().item() / (gn.item() + 1e-30)
            cos = torch.dot(gm, gref_m).item() / (gm.norm().item() * gn.item() + 1e-30)
            print(f"  {precond:>8} {it:5d} {rl:11.3e} {rg:10.3e} {cos:11.6f} {time.time()-t:6.1f}s", flush=True)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
