#!/usr/bin/env python3
"""Break down GlobalRouter.load time into phases to target the load bottleneck."""
import os, sys, time
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root); os.chdir(_root)
import torch

rrg_path = "data/rrg_xcvu3p_int.pt"
from src.router.net_index import NetIndex, default_net_index_path
from src.load_design import load_rrg_fast
from src.router.global_router import GlobalRouter

ni_path = default_net_index_path("./data/", "boom_soc_v2", rrg_path, "directed", 0, 0.1,
                                 route_filter="stubs", edge_scope="corridor",
                                 corridor_width=2, max_edges_per_net=50000)
dev = torch.device("cpu")

t = time.time(); rrg, drows, dcols, c2i, fmt = load_rrg_fast(rrg_path, edge_mode="directed", device=dev)
print(f"[1] load_rrg_fast: {time.time()-t:.1f}s (fmt={fmt})", flush=True)

t = time.time(); ni = NetIndex.load(ni_path)
print(f"[2] NetIndex.load (1.9GB, list of {ni.num_nets} tensors): {time.time()-t:.1f}s", flush=True)

router = GlobalRouter(); router.device = dev; router.edge_mode = "directed"
router.rrg = rrg; router.device_rows = drows; router.device_cols = dcols; router.coord_to_int = c2i
t = time.time(); router._build_int_tile_prefix_sum()
print(f"[3] _build_int_tile_prefix_sum: {time.time()-t:.1f}s", flush=True)

t = time.time(); ni.apply_to_router(router, design=None)
print(f"[4] apply_to_router (274K .tolist/.to): {time.time()-t:.1f}s", flush=True)

router._phys_capacity_tensor = rrg._phys_capacity_tensor.to(dev)
t = time.time(); router._init_edge_weight_tensors()   # calls _build_flat_arrays
print(f"[5] _init_edge_weight_tensors / _build_flat_arrays (274K unique loop): {time.time()-t:.1f}s", flush=True)
print(f"    num_vars={router.num_vars}", flush=True)
print("[done]", flush=True)
