"""Effective-resistance connectivity loss with optional CG solver."""

from typing import List, Tuple

import torch


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
