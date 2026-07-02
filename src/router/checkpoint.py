"""Save/load optimization checkpoints."""

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    x: torch.Tensor,
    lam: torch.Tensor,
    rho: float,
    total_iter: int,
    outer: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "x": x.detach().cpu(),
        "lam": lam.detach().cpu(),
        "rho": rho,
        "total_iter": total_iter,
        "outer": outer,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    data = torch.load(path, map_location=device)
    data["x"] = data["x"].to(device).requires_grad_(True)
    data["lam"] = data["lam"].to(device)
    return data
