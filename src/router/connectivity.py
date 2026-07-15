"""Effective-resistance connectivity loss.

Two implementations:

1. Legacy per-net dense path (`effective_resistance_loss_for_net*`): builds an
   explicit Laplacian per net and differentiates through the solve. Exact but
   O(n^2) memory / O(n^3) time per net -- only usable for small nets/tests.

2. Batched implicit path (`effective_resistance_loss_batched`): one CG solve
   over the block-diagonal Laplacian of *all* nets at once, using sparse
   matvecs on flattened edge arrays. Gradients come from the closed form
       d/dw_e [ b^T L^{-1} b ] = -sum_k (z_k[u_e] - z_k[v_e])^2,   z_k = L^{-1} b_k
   (implicit differentiation), so no autograd graph is built through the
   solver and memory stays O(num_vars + num_local_nodes * col_chunk).
"""

from typing import List, Optional, Tuple

import torch

# Profiling-only knobs (0/False = disabled, normal behaviour). When
# PROFILE_MAX_COL_CHUNKS > 0 the batched forward processes only that many
# column-chunks and returns a partial loss -- used to measure the per-chunk
# time and peak memory of the connectivity solve without running all chunks.
PROFILE_MAX_COL_CHUNKS = 0
PROFILE_VERBOSE = False


def conjugate_gradient_solve(
    L: torch.Tensor,
    b: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Differentiable CG for L x = b (L should be SPD, e.g. L + eps*I)."""
    n = L.shape[0]
    x = torch.zeros(n, device=L.device, dtype=L.dtype)
    r = b - L @ x
    p = r.clone()
    rs_old = torch.dot(r, r)
    if rs_old < tol * tol:
        return x
    for _ in range(max_iter):
        Ap = L @ p
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new
    return x


def effective_resistance_loss_for_net(
    x_slice: torch.Tensor,
    edge_list: List[int],
    directed_edges: List[Tuple[int, int]],
    src: int,
    sinks: List[int],
    device: torch.device,
    eps: float = 1e-6,
    solver: str = "solve",
    cg_max_iter: int = 50,
) -> torch.Tensor:
    """Sum effective resistance from source to each sink on net subgraph."""
    if torch.is_tensor(edge_list):
        edge_list = edge_list.tolist()
    nodes_set = set()
    for de_idx in edge_list:
        u, v = directed_edges[de_idx]
        nodes_set.add(u)
        nodes_set.add(v)
    nodes_list = sorted(nodes_set)
    global_to_local = {g: loc for loc, g in enumerate(nodes_list)}
    n = len(nodes_list)

    if src not in global_to_local or n < 2:
        return torch.tensor(0.0, device=device, dtype=x_slice.dtype)

    L = torch.zeros(n, n, device=device, dtype=x_slice.dtype)
    for k, de_idx in enumerate(edge_list):
        u, v = directed_edges[de_idx]
        if u not in global_to_local or v not in global_to_local:
            continue
        iu, iv = global_to_local[u], global_to_local[v]
        w = x_slice[k] + 1e-8
        L[iu, iv] = L[iu, iv] - w
        L[iv, iu] = L[iv, iu] - w
        L[iu, iu] = L[iu, iu] + w
        L[iv, iv] = L[iv, iv] + w

    L_reg = L + eps * torch.eye(n, device=device, dtype=x_slice.dtype)
    loss = torch.tensor(0.0, device=device, dtype=x_slice.dtype)

    for sink in sinks:
        if sink not in global_to_local:
            continue
        local_src = global_to_local[src]
        local_sink = global_to_local[sink]
        b = torch.zeros(n, device=device, dtype=x_slice.dtype)
        b[local_src] = 1.0
        b[local_sink] = -1.0

        if solver == "cg":
            z = conjugate_gradient_solve(L_reg, b, max_iter=cg_max_iter)
        else:
            z = torch.linalg.solve(L_reg, b.unsqueeze(1)).squeeze(1)
        loss = loss + (b * z).sum()
    return loss


def _laplacian_matvec(
    w: torch.Tensor,
    flat_u: torch.Tensor,
    flat_v: torch.Tensor,
    V: torch.Tensor,
    eps: float,
    edge_chunk: int = 8_000_000,
) -> torch.Tensor:
    """(L + eps*I) @ V for the block-diagonal Laplacian over all nets.

    w: [E] edge weights; flat_u/flat_v: [E] local node indices; V: [n, C].

    The per-edge difference `duv` is the dominant transient: shape [E, C], which
    for large designs (E ~ 2e8) is tens of GB even at modest C. `edge_chunk > 0`
    accumulates the matvec in slices of `edge_chunk` edges so the temporary is
    bounded to [edge_chunk, C] -- identical result, O(edge_chunk*C) peak memory.
    """
    out = V * eps
    E = flat_u.shape[0]
    if edge_chunk <= 0 or edge_chunk >= E:
        duv = V.index_select(0, flat_u) - V.index_select(0, flat_v)
        duv.mul_(w.unsqueeze(1))
        out.index_add_(0, flat_u, duv)
        duv.neg_()
        out.index_add_(0, flat_v, duv)
        return out
    for s in range(0, E, edge_chunk):
        e = min(s + edge_chunk, E)
        fu = flat_u[s:e]
        fv = flat_v[s:e]
        duv = V.index_select(0, fu) - V.index_select(0, fv)
        duv.mul_(w[s:e].unsqueeze(1))
        out.index_add_(0, fu, duv)
        duv.neg_()
        out.index_add_(0, fv, duv)
    return out


def _cg_block(
    w: torch.Tensor,
    flat_u: torch.Tensor,
    flat_v: torch.Tensor,
    B: torch.Tensor,
    eps: float,
    max_iter: int,
    tol: float,
    edge_chunk: int = 8_000_000,
) -> torch.Tensor:
    """Batched CG for (L + eps*I) Z = B with per-column convergence tracking."""
    X = torch.zeros_like(B)
    R = B.clone()
    P = R.clone()
    rs = (R * R).sum(dim=0)
    thresh = (tol * tol) * rs.clamp_min(1e-30)
    for _ in range(max_iter):
        if bool((rs <= thresh).all()):
            break
        AP = _laplacian_matvec(w, flat_u, flat_v, P, eps, edge_chunk=edge_chunk)
        pAp = (P * AP).sum(dim=0)
        active = rs > thresh
        alpha = torch.where(active, rs / pAp.clamp_min(1e-30), torch.zeros_like(rs))
        X += P * alpha.unsqueeze(0)
        R -= AP * alpha.unsqueeze(0)
        rs_new = (R * R).sum(dim=0)
        beta = torch.where(active, rs_new / rs.clamp_min(1e-30), torch.zeros_like(rs))
        P = R + P * beta.unsqueeze(0)
        rs = rs_new
    return X


class BatchedEffectiveResistance(torch.autograd.Function):
    """Sum of source->sink effective resistances over all nets, implicit grad.

    Each (net, sink) pair occupies one column of the RHS. Because net blocks
    are disjoint in the flattened local-node space, sinks of *different* nets
    can share a column; columns are processed in chunks of `col_chunk`.
    """

    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,
        flat_u: torch.Tensor,
        flat_v: torch.Tensor,
        src_flat: torch.Tensor,
        sink_flat: torch.Tensor,
        col_id: torch.Tensor,
        num_nodes: int,
        num_cols: int,
        eps: float,
        col_chunk: int,
        cg_max_iter: int,
        cg_tol: float,
        edge_chunk: int,
    ) -> torch.Tensor:
        device, dtype = w.device, w.dtype
        wd = w.detach()
        grad_w = torch.zeros_like(wd)
        loss = torch.zeros((), device=device, dtype=dtype)
        E = flat_u.shape[0]
        _prof_n = 0
        for c0 in range(0, num_cols, col_chunk):
            if PROFILE_MAX_COL_CHUNKS and _prof_n >= PROFILE_MAX_COL_CHUNKS:
                break
            _prof_n += 1
            c1 = min(c0 + col_chunk, num_cols)
            sel = ((col_id >= c0) & (col_id < c1)).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            cols = col_id[sel] - c0
            ones = torch.ones(sel.numel(), device=device, dtype=dtype)
            B = torch.zeros(num_nodes, c1 - c0, device=device, dtype=dtype)
            B.index_put_((src_flat[sel], cols), ones, accumulate=True)
            B.index_put_((sink_flat[sel], cols), -ones, accumulate=True)
            Z = _cg_block(wd, flat_u, flat_v, B, eps, cg_max_iter, cg_tol,
                          edge_chunk=edge_chunk)
            loss = loss + (B * Z).sum()
            # grad_w[e] -= sum_c (Z[u_e,c] - Z[v_e,c])^2. The [E, C] difference is
            # the same transient as the matvec, so accumulate it in edge slices too.
            if edge_chunk <= 0 or edge_chunk >= E:
                zu = Z.index_select(0, flat_u)
                zv = Z.index_select(0, flat_v)
                grad_w -= (zu - zv).square().sum(dim=1)
            else:
                for s in range(0, E, edge_chunk):
                    e = min(s + edge_chunk, E)
                    zu = Z.index_select(0, flat_u[s:e])
                    zv = Z.index_select(0, flat_v[s:e])
                    grad_w[s:e] -= (zu - zv).square().sum(dim=1)
            del B, Z
        ctx.save_for_backward(grad_w)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        (grad_w,) = ctx.saved_tensors
        return (grad_out * grad_w,) + (None,) * 12


def effective_resistance_loss_batched(
    x: torch.Tensor,
    conn: Optional[dict],
    eps: float = 1e-6,
    cg_max_iter: int = 100,
    cg_tol: float = 1e-5,
    col_chunk: int = 8,
    weight_floor: float = 1e-8,
    edge_chunk: int = 8_000_000,
) -> torch.Tensor:
    """All-nets effective resistance loss via batched CG + implicit diff.

    `conn` is the precomputed flat structure from
    GlobalRouter._build_flat_arrays (or None if no net has usable terminals).
    """
    if conn is None or conn["num_cols"] == 0:
        return x.sum() * 0.0
    w = x + weight_floor
    return BatchedEffectiveResistance.apply(
        w,
        conn["flat_u"],
        conn["flat_v"],
        conn["src_flat"],
        conn["sink_flat"],
        conn["col_id"],
        conn["num_nodes"],
        conn["num_cols"],
        eps,
        col_chunk,
        cg_max_iter,
        cg_tol,
        edge_chunk,
    )


def effective_resistance_loss_for_net_undirected(
    x_slice: torch.Tensor,
    phys_edge_list: List[int],
    phys_endpoints_fn,
    src: int,
    sinks: List[int],
    device: torch.device,
    eps: float = 1e-6,
    solver: str = "solve",
    cg_max_iter: int = 50,
) -> torch.Tensor:
    """Sum effective resistance from source to each sink using undirected phys edges."""
    if torch.is_tensor(phys_edge_list):
        phys_edge_list = phys_edge_list.tolist()
    nodes_set = set()
    for phys_id in phys_edge_list:
        u, v = phys_endpoints_fn(phys_id)
        nodes_set.add(u)
        nodes_set.add(v)
    nodes_list = sorted(nodes_set)
    global_to_local = {g: loc for loc, g in enumerate(nodes_list)}
    n = len(nodes_list)

    if src not in global_to_local or n < 2:
        return torch.tensor(0.0, device=device, dtype=x_slice.dtype)

    L = torch.zeros(n, n, device=device, dtype=x_slice.dtype)
    for k, phys_id in enumerate(phys_edge_list):
        u, v = phys_endpoints_fn(phys_id)
        if u not in global_to_local or v not in global_to_local:
            continue
        iu, iv = global_to_local[u], global_to_local[v]
        w = x_slice[k] + 1e-8
        L[iu, iv] = L[iu, iv] - w
        L[iv, iu] = L[iv, iu] - w
        L[iu, iu] = L[iu, iu] + w
        L[iv, iv] = L[iv, iv] + w

    L_reg = L + eps * torch.eye(n, device=device, dtype=x_slice.dtype)
    loss = torch.tensor(0.0, device=device, dtype=x_slice.dtype)

    for sink in sinks:
        if sink not in global_to_local:
            continue
        local_src = global_to_local[src]
        local_sink = global_to_local[sink]
        b = torch.zeros(n, device=device, dtype=x_slice.dtype)
        b[local_src] = 1.0
        b[local_sink] = -1.0

        if solver == "cg":
            z = conjugate_gradient_solve(L_reg, b, max_iter=cg_max_iter)
        else:
            z = torch.linalg.solve(L_reg, b.unsqueeze(1)).squeeze(1)
        loss = loss + (b * z).sum()
    return loss
