"""Augmented Lagrangian optimizer for GlobalRouter."""

import os
from typing import Optional

import torch

from src.router.checkpoint import save_checkpoint
from src.router.meng_lambda import update_multipliers_meng


def optimize_augmented_lagrangian(
    router,
    x: Optional[torch.Tensor] = None,
    num_outer: int = 200,
    num_inner: int = 5,
    max_iterations: int = 1000,
    viz_interval: int = 50,
    viz_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    save_every: int = 0,
    resume_path: Optional[str] = None,
    lr_x: float = 0.01,
    lr_lam: float = 0.1,
    rho: float = 1.0,
    rho_max: float = 100.0,
    rho_mult: float = 2.0,
    w_wl: float = 1.0,
    w_conn: float = 1.0,
    w_flow: float = 1.0,
    w_disc: float = 0.0,
    disc_ramp_outer: int = 0,
    conn_every: int = 1,
    conn_freeze_outer: int = 0,
    early_stop_tol: float = 0.0,
    early_stop_patience: int = 3,
    connectivity: str = "effective_resistance",
    connectivity_solver: str = "cg",
    conn_net_batch: int = 0,
    flow_net_batch: int = 0,
    verbose: bool = True,
    log_setup: bool = True,
) -> torch.Tensor:
    router.connectivity_solver = connectivity_solver
    router.conn_net_batch = conn_net_batch
    router.flow_net_batch = flow_net_batch

    if log_setup:
        print("  [4a] Initializing x variables...")
    overflow_ref = None
    if resume_path and os.path.isfile(resume_path):
        from src.router.checkpoint import load_checkpoint
        ckpt = load_checkpoint(resume_path, router.device)
        x = ckpt["x"]
        lam = ckpt["lam"]
        rho = ckpt.get("rho", rho)
        total_iter = ckpt.get("total_iter", 0)
        start_outer = ckpt.get("outer", 0)
        if "overflow_ref" in ckpt:
            overflow_ref = ckpt["overflow_ref"].to(router.device)
        if log_setup:
            print(f"  [4a] Resumed from {resume_path} at iter {total_iter}")
    else:
        if x is None:
            x = router.init_variables()
        else:
            x = x.detach().clone().requires_grad_(True)
        lam = torch.zeros(len(router.rrg.phys_list), device=router.device, dtype=x.dtype)
        total_iter = 0
        start_outer = 0

    if not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)

    if log_setup:
        print(f"  [4b] Created λ multipliers: {len(router.rrg.phys_list)} physical edges")

    optimizer = torch.optim.Adam([x], lr=lr_x)
    if log_setup:
        print(f"  [4c] Adam optimizer ready (lr_x={lr_x})")

    gif_frames = []
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
        if log_setup:
            print(f"  [4d] Viz dir ready: {viz_dir}")

    if log_setup:
        print("  [4e] Starting outer loop (Augmented Lagrangian)...")
        if early_stop_tol > 0:
            print(
                f"  [4e] Early stop armed: overflow improvement < "
                f"{early_stop_tol*100:.1f}% over {early_stop_patience} outers "
                f"(after rho reaches {rho_max})"
            )

    ovf_hist: list = []
    for outer in range(start_outer, num_outer):
        # Anneal discretization weight from 0 to w_disc over disc_ramp_outer outers
        # (establish connectivity first, then sharpen to discrete paths).
        if w_disc != 0.0 and disc_ramp_outer > 0:
            w_disc_eff = w_disc * min(1.0, outer / float(disc_ramp_outer))
        else:
            w_disc_eff = w_disc
        # A2: freeze connectivity after it has converged (conn_freeze_outer>0).
        conn_frozen = conn_freeze_outer > 0 and outer >= conn_freeze_outer
        for _inner in range(num_inner):
            optimizer.zero_grad()
            # A1: evaluate the (expensive) connectivity term only every conn_every
            # inner steps; other steps optimize wirelength/congestion/flow only.
            include_conn = (not conn_frozen) and (total_iter % conn_every == 0)
            w_conn_step = w_conn if include_conn else 0.0
            L_A = router.augmented_lagrangian(
                x,
                lam,
                rho,
                w_wl=w_wl,
                w_conn=w_conn_step,
                w_flow=w_flow,
                w_disc=w_disc_eff,
                connectivity=connectivity,
                connectivity_solver=connectivity_solver,
                conn_net_batch=conn_net_batch,
                flow_net_batch=flow_net_batch,
            )
            L_A.backward()
            optimizer.step()
            with torch.no_grad():
                x.clamp_(0.0, 1.0)

            total_iter += 1

            if viz_dir and total_iter % viz_interval == 0:
                if verbose:
                    print(f"      [viz] Capturing frame at iter {total_iter}")
                from PIL import Image
                from src.Visualizer import render_congestion_frame
                cong_grid = router.get_congestion_map(x)
                frame_arr = render_congestion_frame(
                    cong_grid,
                    title=f"Congestion (flow/capacity) @ iter {total_iter}",
                )
                gif_frames.append(Image.fromarray(frame_arr))

            if checkpoint_dir and save_every > 0 and total_iter % save_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_iter{total_iter:06d}.pt")
                extra = {"overflow_ref": overflow_ref.detach().cpu()} if overflow_ref is not None else None
                save_checkpoint(ckpt_path, x, lam, rho, total_iter, outer + 1, extra=extra)

            if total_iter >= max_iterations:
                if verbose:
                    print(f"      [stop] Reached max_iterations={max_iterations}")
                break

        if total_iter >= max_iterations:
            break

        with torch.no_grad():
            _, overflows = router._get_usage_and_overflows(x)
            if overflow_ref is None:
                overflow_ref = torch.clamp(overflows.clone(), min=1e-8)
            lam = update_multipliers_meng(lam, overflows, overflow_ref, lr_lam)

        if overflows.max().item() > 0.1 and rho < rho_max:
            rho = min(rho * rho_mult, rho_max)
            if verbose:
                print(f"      [rho] Increased to {rho:.2f}")

        # Early stop on AL convergence. Overflow (the constraint violation) is already
        # computed above for the lambda update, so this costs nothing. A fixed
        # max-iterations is design-size-dependent: it over-runs small designs and
        # under-runs large ones. Stopping when overflow stops improving adapts to both.
        #
        # Only arm once rho has reached rho_max: overflow legitimately RISES while the
        # penalty ramps (flow spreads before it is squeezed), so an earlier check would
        # fire on that transient.
        if early_stop_tol > 0:
            ovf_hist.append(float(overflows.sum().item()))
            if rho >= rho_max and len(ovf_hist) > early_stop_patience:
                window = ovf_hist[-(early_stop_patience + 1):]
                base = max(window[0], 1e-12)
                improvement = (window[0] - window[-1]) / base
                if improvement < early_stop_tol:
                    if verbose:
                        print(
                            f"      [early-stop] outer {outer+1} (iter {total_iter}): "
                            f"overflow improved {improvement*100:.2f}% over last "
                            f"{early_stop_patience} outers (< {early_stop_tol*100:.1f}%); "
                            f"overflow_sum={window[-1]:.4g}"
                        )
                    break

        if verbose and (outer + 1) % 20 == 0:
            wl = router.wirelength_loss(x).item()
            conn = router.connectivity_loss_effective_resistance(
                x, solver=connectivity_solver, net_batch=conn_net_batch
            ).item()
            max_of = overflows.max().item()
            lam_min, lam_max = lam.min().item(), lam.max().item()
            lam_mean = lam.float().mean().item()
            lam_norm = lam.norm().item()
            if getattr(router, "edge_mode", "directed") == "undirected":
                print(
                    f"  [metrics] Outer {outer+1}/{num_outer} (iter {total_iter}): "
                    f"wl={wl:.2f} conn={conn:.2f} max_overflow={max_of:.4f} | "
                    f"λ: min={lam_min:.4f} max={lam_max:.4f} mean={lam_mean:.4f} ‖λ‖={lam_norm:.2f}"
                )
            else:
                flow = router.flow_conservation_loss(
                    x, net_batch=flow_net_batch
                ).item()
                print(
                    f"  [metrics] Outer {outer+1}/{num_outer} (iter {total_iter}): "
                    f"wl={wl:.2f} conn={conn:.2f} flow={flow:.2f} max_overflow={max_of:.4f} | "
                    f"λ: min={lam_min:.4f} max={lam_max:.4f} mean={lam_mean:.4f} ‖λ‖={lam_norm:.2f}"
                )

    if log_setup:
        print("  [4f] Optimization loop finished")

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_ckpt = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        extra = {"overflow_ref": overflow_ref.detach().cpu()} if overflow_ref is not None else None
        save_checkpoint(final_ckpt, x, lam, rho, total_iter, outer + 1, extra=extra)
        global_x_path = os.path.join(checkpoint_dir, "global_x.pt")
        torch.save(x.detach().cpu(), global_x_path)
        if log_setup:
            print(f"  [4f] Saved checkpoint: {final_ckpt}")

    if viz_dir and gif_frames:
        if log_setup:
            print(f"  [4g] Writing congestion evolution GIF ({len(gif_frames)} frames)...")
        gif_path = os.path.join(viz_dir, "congestion_evolution.gif")
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=200,
            loop=0,
        )
        if log_setup:
            print(f"  [4g] Saved: {gif_path}")

    return x.detach()
