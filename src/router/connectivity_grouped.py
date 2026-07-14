"""Prototype: net-grouped effective-resistance connectivity loss.

The production `effective_resistance_loss_batched` chunks the RHS by *columns*
and runs each chunk's CG matvec over the ENTIRE flattened edge array (all nets).
Because net blocks are disjoint, a 32-column chunk only involves ~16-32 nets, so
>99.99% of every matvec multiplies zeros -- making one forward ~8 days on
boom_soc_v2 (42,672 chunks x full-graph matvec).

This prototype instead groups *consecutive nets* (they are laid out contiguously
in the flat arrays) and runs each group's CG on only that group's subgraph
(edges + nodes sliced to the group). Work drops from O(E x num_chunks) to
~O(E) per CG sweep. Math is identical -- validated against the batched impl.

Not yet wired into production; used by scripts/profile_gpu_mem.py to benchmark.
"""

from typing import List, Optional, Tuple

import torch


def _build_groups(
    net_ncol: torch.Tensor,   # [num_nets] columns contributed by each net
    col_chunk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy: pack consecutive nets into groups of <= col_chunk columns.

    A single net with > col_chunk columns becomes its own (oversized) group.
    Returns (group_net_start, group_net_end) as 1-D tensors over group index,
    where net indices [start, end) form each group.
    """
    ncol = net_ncol.tolist()
    starts: List[int] = []
    ends: List[int] = []
    i = 0
    n = len(ncol)
    while i < n:
        c = 0
        j = i
        while j < n and (c + ncol[j] <= col_chunk or j == i):
            c += ncol[j]
            j += 1
        starts.append(i)
        ends.append(j)
        i = j
    return (torch.tensor(starts, dtype=torch.long),
            torch.tensor(ends, dtype=torch.long))


def _cg_block(w, flat_u, flat_v, B, eps, max_iter, tol, num_nodes,
              precond="none", X0=None):
    """Batched CG for (L + eps*I) Z = B on a single group's subgraph.

    precond="jacobi" applies diagonal (Jacobi) preconditioning: M = diag(L+eps*I),
    which is eps + (sum of incident edge weights) per node. Near-free per iteration
    and dramatically cuts iterations when edge weights span a wide range.

    X0 (optional [nn, ncols]) is a warm-start initial guess -- reusing the previous
    AL iteration's solution makes CG re-converge in far fewer iterations because x
    (hence the solution) moves slowly between steps.
    """
    def matvec(V):
        duv = V.index_select(0, flat_u) - V.index_select(0, flat_v)
        duv.mul_(w.unsqueeze(1))
        out = V * eps
        out.index_add_(0, flat_u, duv)
        duv.neg_()
        out.index_add_(0, flat_v, duv)
        return out

    if precond == "jacobi":
        d = torch.full((num_nodes,), eps, device=B.device, dtype=B.dtype)
        d.index_add_(0, flat_u, w)
        d.index_add_(0, flat_v, w)
        Minv = (1.0 / d.clamp_min(1e-30)).unsqueeze(1)   # [nn, 1]
    else:
        Minv = None

    if X0 is None:
        X = torch.zeros_like(B)
        R = B.clone()
    else:
        X = X0.clone()
        R = B - matvec(X)
    Z = R * Minv if Minv is not None else R
    P = Z.clone()
    rz = (R * Z).sum(dim=0)
    rs = (R * R).sum(dim=0)
    thresh = (tol * tol) * rs.clamp_min(1e-30)
    for _ in range(max_iter):
        if bool((rs <= thresh).all()):
            break
        AP = matvec(P)
        pAp = (P * AP).sum(dim=0)
        active = rs > thresh
        alpha = torch.where(active, rz / pAp.clamp_min(1e-30), torch.zeros_like(rz))
        X += P * alpha.unsqueeze(0)
        R -= AP * alpha.unsqueeze(0)
        Znew = R * Minv if Minv is not None else R
        rz_new = (R * Znew).sum(dim=0)
        beta = torch.where(active, rz_new / rz.clamp_min(1e-30), torch.zeros_like(rz))
        P = Znew + P * beta.unsqueeze(0)
        rz = rz_new
        rs = (R * R).sum(dim=0)
    return X


class GroupedEffectiveResistance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w, flat_u, flat_v, src_flat, sink_flat, col_id,
        var_offset,   # [num_nets+1] edge offsets per net
        node_offset,  # [num_nets+1] local-node offsets per net
        net_c_off,    # [num_nets+1] column offsets per net (cumsum of net_ncol)
        g_start, g_end,  # group -> [net_start, net_end)
        eps, cg_max_iter, cg_tol,
        max_groups,   # profiling: process only this many groups (0 = all)
        precond,      # "none" | "jacobi"
        ws_cache,     # None, or dict{group_idx -> Z} for warm-start across calls
    ):
        device, dtype = w.device, w.dtype
        wd = w.detach()
        grad_w = torch.zeros_like(wd)
        loss = torch.zeros((), device=device, dtype=dtype)
        ng = int(g_start.numel())
        limit = ng if max_groups <= 0 else min(ng, max_groups)
        vo = var_offset.tolist()
        no = node_offset.tolist()
        co = net_c_off.tolist()
        gs = g_start.tolist()
        ge = g_end.tolist()
        for g in range(limit):
            a, b = gs[g], ge[g]
            e0, e1 = vo[a], vo[b]
            n0, n1 = no[a], no[b]
            c0, c1 = co[a], co[b]
            if e1 == e0 or c1 == c0:
                continue
            fu = flat_u[e0:e1].long() - n0
            fv = flat_v[e0:e1].long() - n0
            wg = wd[e0:e1]
            ncols = c1 - c0
            nn = n1 - n0
            cols = (col_id[c0:c1].long() - c0)
            sv = src_flat[c0:c1].long() - n0
            kv = sink_flat[c0:c1].long() - n0
            ones = torch.ones(ncols, device=device, dtype=dtype)
            B = torch.zeros(nn, ncols, device=device, dtype=dtype)
            B.index_put_((sv, cols), ones, accumulate=True)
            B.index_put_((kv, cols), -ones, accumulate=True)
            X0 = ws_cache.get(g) if ws_cache is not None else None
            Z = _cg_block(wg, fu, fv, B, eps, cg_max_iter, cg_tol, nn,
                          precond=precond, X0=X0)
            if ws_cache is not None:
                ws_cache[g] = Z.detach()
            loss = loss + (B * Z).sum()
            zu = Z.index_select(0, fu)
            zv = Z.index_select(0, fv)
            grad_w[e0:e1] -= (zu - zv).square().sum(dim=1)
            del B, Z
        ctx.save_for_backward(grad_w)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        (grad_w,) = ctx.saved_tensors
        return (grad_out * grad_w,) + (None,) * 17


def effective_resistance_loss_grouped(
    x: torch.Tensor,
    conn: Optional[dict],
    var_offset: torch.Tensor,
    node_offset: torch.Tensor,
    eps: float = 1e-6,
    cg_max_iter: int = 100,
    cg_tol: float = 1e-5,
    col_chunk: int = 32,
    weight_floor: float = 1e-8,
    max_groups: int = 0,
    precond: str = "none",
    ws_cache: Optional[dict] = None,
    _group_cache: Optional[dict] = None,
) -> torch.Tensor:
    """Net-grouped effective-resistance loss. Same result as the batched impl.

    var_offset/node_offset are the router's per-net edge/node offsets
    (`_var_offset`, `_node_offset`) as long tensors of length num_nets+1.
    """
    if conn is None or conn["num_cols"] == 0:
        return x.sum() * 0.0
    w = x + weight_floor
    device = x.device

    src_flat = conn["src_flat"]
    node_offset = node_offset.to(device)
    var_offset = var_offset.to(device)

    # Per-net column count: each column's src node lies in exactly one net block.
    # node_offset is sorted, so searchsorted maps a node id -> its net index.
    col_net = torch.searchsorted(node_offset, src_flat.long(), right=True) - 1
    num_nets = int(var_offset.numel() - 1)
    net_ncol = torch.bincount(col_net, minlength=num_nets)
    net_c_off = torch.zeros(num_nets + 1, dtype=torch.long, device=device)
    net_c_off[1:] = torch.cumsum(net_ncol, dim=0)

    if _group_cache is not None and _group_cache.get("col_chunk") == col_chunk:
        g_start = _group_cache["g_start"]
        g_end = _group_cache["g_end"]
    else:
        g_start, g_end = _build_groups(net_ncol.cpu(), col_chunk)
        g_start = g_start.to(device)
        g_end = g_end.to(device)
        if _group_cache is not None:
            _group_cache["col_chunk"] = col_chunk
            _group_cache["g_start"] = g_start
            _group_cache["g_end"] = g_end
            _group_cache["num_groups"] = int(g_start.numel())

    return GroupedEffectiveResistance.apply(
        w, conn["flat_u"], conn["flat_v"], src_flat,
        conn["sink_flat"], conn["col_id"],
        var_offset, node_offset, net_c_off,
        g_start, g_end,
        eps, cg_max_iter, cg_tol, max_groups, precond, ws_cache,
    )
