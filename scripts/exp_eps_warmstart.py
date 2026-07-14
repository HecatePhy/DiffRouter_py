#!/usr/bin/env python3
"""Two conditioning experiments on one loaded router (boom_soc_v2, hard x):

  PART 1  larger-eps: does raising the Laplacian regularization eps make CG
          converge in few iters? And does the larger-eps gradient still point
          the same way as the true (eps=1e-6) gradient (bias check)?

  PART 2  warm-start: on real hard groups, does initializing CG from the
          previous x's solution cut iterations-to-converge vs a cold start?
"""
import argparse, os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--col-chunk", type=int, default=128)
    ap.add_argument("--sample-groups", type=int, default=1500)
    ap.add_argument("--ref-iters", type=int, default=400)
    args = ap.parse_args()

    from src.router.global_router import GlobalRouter
    from src.router.net_index import default_net_index_path
    from src.router.connectivity_grouped import effective_resistance_loss_grouped, _cg_block

    dev = torch.device(f"cuda:{args.gpu}")
    ni = default_net_index_path("./data/", "boom_soc_v2", "data/rrg_xcvu3p_int.pt",
                                "directed", 0, 0.1, route_filter="stubs",
                                edge_scope="corridor", corridor_width=2, max_edges_per_net=50000)
    t0 = time.time()
    router = GlobalRouter.load("data/rrg_xcvu3p_int.pt", ni, device=dev,
                               edge_mode="directed", verbose=True)
    print(f"[load] {time.time()-t0:.1f}s  nets={router.num_nets}", flush=True)

    conn = router._conn
    vo = torch.tensor(router._var_offset, dtype=torch.long, device=dev)
    no = torch.tensor(router._node_offset, dtype=torch.long, device=dev)
    gcache = {}
    idx = torch.arange(router.num_vars, device=dev, dtype=torch.float32)
    x_hard = (0.5 + 0.49 * torch.sin(idx * 0.7)).clamp_(0.02, 0.98)

    def grad_at(eps, iters, tol=1e-13):
        xg = x_hard.clone().requires_grad_(True)
        L = effective_resistance_loss_grouped(xg, conn, vo, no, eps=eps,
                cg_max_iter=iters, cg_tol=tol, col_chunk=args.col_chunk,
                max_groups=args.sample_groups, _group_cache=gcache)
        L.backward()
        return L.item(), xg.grad.detach()

    def cos(a, b):
        return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-30)

    print("\n===== PART 1: larger eps =====", flush=True)
    refs = {}
    for eps in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        t = time.time()
        Lr, gr = grad_at(eps, args.ref_iters)
        refs[eps] = gr
        print(f"[eps={eps:.0e}] converged ref ({args.ref_iters} it, {time.time()-t:.0f}s): "
              f"loss={Lr:.4e}", flush=True)
        m = gr != 0
        for it in (50, 20, 10, 5):
            L, g = grad_at(eps, it)
            print(f"     it={it:3d}  gradCosSim(vs same-eps ref)={cos(g[m], gr[m]):.6f}", flush=True)
    # bias: does large-eps gradient point like the true eps=1e-6 gradient?
    print("\n[bias] cosine of converged grad(eps) vs converged grad(eps=1e-6):", flush=True)
    base = refs[1e-6]; m = base != 0
    for eps in (1e-4, 1e-3, 1e-2, 1e-1):
        print(f"     eps={eps:.0e}: {cos(refs[eps][m], base[m]):.6f}", flush=True)

    print("\n===== PART 2: warm-start (real hard groups) =====", flush=True)
    # Rebuild the group table, pick the largest few groups (most edges = hardest).
    _ = effective_resistance_loss_grouped(x_hard.detach(), conn, vo, no, eps=1e-6,
            cg_max_iter=1, cg_tol=1e-13, col_chunk=args.col_chunk, max_groups=1,
            _group_cache=gcache)
    gs = gcache["g_start"].tolist(); ge = gcache["g_end"].tolist()
    vol = vo.tolist(); nol = no.tolist()
    # column offsets per net
    col_net = torch.searchsorted(no, conn["src_flat"].long(), right=True) - 1
    net_ncol = torch.bincount(col_net, minlength=router.num_nets)
    net_c_off = torch.zeros(router.num_nets + 1, dtype=torch.long, device=dev)
    net_c_off[1:] = torch.cumsum(net_ncol, 0)
    cofl = net_c_off.tolist()
    sizes = [(vol[ge[i]] - vol[gs[i]], i) for i in range(len(gs))]
    sizes.sort(reverse=True)
    picks = [i for _, i in sizes[:5]]

    eps = 1e-6
    for g in picks:
        a, b = gs[g], ge[g]
        e0, e1 = vol[a], vol[b]; n0, n1 = nol[a], nol[b]; c0, c1 = cofl[a], cofl[b]
        fu = conn["flat_u"][e0:e1].long() - n0
        fv = conn["flat_v"][e0:e1].long() - n0
        nn = n1 - n0; ncols = c1 - c0
        cols = conn["col_id"][c0:c1].long() - c0
        sv = conn["src_flat"][c0:c1].long() - n0
        kv = conn["sink_flat"][c0:c1].long() - n0
        def buildB():
            B = torch.zeros(nn, ncols, device=dev, dtype=torch.float32)
            o = torch.ones(ncols, device=dev)
            B.index_put_((sv, cols), o, accumulate=True)
            B.index_put_((kv, cols), -o, accumulate=True)
            return B
        B = buildB()
        wa = (x_hard[e0:e1] + 1e-8)
        # converged solution at x_a
        Za = _cg_block(wa, fu, fv, B, eps, 2000, 1e-13, nn)
        # simulate an optimizer step: x_b = x_a nudged
        step = 0.02 * torch.cos(torch.arange(e1 - e0, device=dev) * 0.9)
        wb = (x_hard[e0:e1] + step).clamp(0.02, 0.98) + 1e-8
        Zb_true = _cg_block(wb, fu, fv, B, eps, 4000, 1e-14, nn)
        def relerr(Z):
            return (Z - Zb_true).norm().item() / (Zb_true.norm().item() + 1e-30)
        print(f"\n  group {g}: edges={e1-e0}  nodes={nn}  cols={ncols}", flush=True)
        for it in (10, 20, 40):
            Zc = _cg_block(wb, fu, fv, B, eps, it, 1e-14, nn)                 # cold
            Zw = _cg_block(wb, fu, fv, B, eps, it, 1e-14, nn, X0=Za)          # warm
            print(f"    it={it:3d}  cold relerr={relerr(Zc):.3e}   warm relerr={relerr(Zw):.3e}", flush=True)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
