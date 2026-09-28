"""Jointly sample four-camera rigs for a measured overlap curriculum."""
from __future__ import annotations

import numpy as np
from infinigen.p4d import cameras as C

TIERS = ("high", "medium", "low")


def preview_overlap(views, intrinsics, ray_fn):
    """Coarse surface coverage, used for rejection sampling, never as final GT."""
    T = len(views[0]["pos"])
    scores = []
    for t in sorted(set(np.linspace(0, T - 1, 5).astype(int))):
        coverage = np.full((len(views), len(views)), np.nan)
        for i, (view, intr) in enumerate(zip(views, intrinsics)):
            u, v = np.meshgrid((np.arange(16) + .5) / 16 * intr.width,
                               (np.arange(9) + .5) / 9 * intr.height)
            rays = np.stack([(u.ravel() - intr.cx) / intr.fx, (v.ravel() - intr.cy) / intr.fy,
                             np.ones(u.size)], -1)
            rays = rays @ (view["R"][t] @ np.diag([1, -1, -1])).T
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            origin = view["pos"][t]
            dist = ray_fn(np.broadcast_to(origin, rays.shape), rays)
            ok = np.isfinite(dist) & (dist > 0) & (dist < 1e4)
            if ok.sum() < 16:
                continue
            points = origin + rays[ok] * dist[ok, None]
            for j, (dest, intr2) in enumerate(zip(views, intrinsics)):
                uv, z = C.project(points, dest["pos"][t:t+1], dest["R"][t:t+1], intr2)
                uv, z = uv[0], z[0]
                visible = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < intr2.width) & (
                    uv[:, 1] >= 0) & (uv[:, 1] < intr2.height)
                delta = points[visible] - dest["pos"][t]
                lengths = np.linalg.norm(delta, axis=1)
                hits = ray_fn(np.broadcast_to(dest["pos"][t], delta.shape),
                              delta / np.maximum(lengths[:, None], 1e-12))
                visible[visible] &= np.abs(hits - lengths) <= np.maximum(.02, .01 * lengths)
                coverage[i, j] = visible.mean()
        ij = np.triu_indices(len(views), 1)
        scores.extend(((coverage + coverage.T) / 2)[ij].tolist())
    finite = np.asarray(scores)[np.isfinite(scores)]
    return float(np.median(finite)) if len(finite) else None


def sample_rig(rng, T, fps, intrinsics, anchor_fn, target, tier, clearance_fn, ray_fn,
               inside_fn=None, max_tries=160, view_targets=None):
    """One tier, four distinct paths, one jittered view. Failure is explicit."""
    if tier not in TIERS or len(intrinsics) != 4:
        raise ValueError("a curriculum rig requires a tier and four cameras")
    target = C._target_at(target, T)
    # High-overlap paths share an action, but still have independent motion.
    # A fixed aim loses overlap as the action crosses the frame. An in-place
    # tracking pan keeps its distinct path while following the shared action.
    pool = ("spin", "crane", "dolly", "handheld") if tier == "high" else C.PATH_TYPES
    types = list(rng.choice(pool, 4, replace=False))
    jitter_view = int(rng.integers(4))
    last = None
    for attempt in range(max_tries):
        if attempt and attempt % 20 == 0:
            types = list(rng.choice(pool, 4, replace=False))
        anchor0 = np.asarray(anchor_fn(rng), float)
        scale = max(1.0, np.linalg.norm(anchor0 - target[0]))
        spread = .003 if tier == "high" else .22
        views = []
        try:
            for v in range(4):
                tgt = C._target_at(view_targets[v], T) if view_targets is not None else target
                anchor = (np.asarray(anchor_fn(rng)) if tier == "low" or (tier == "medium" and v >= 2) else
                          anchor0 + rng.normal(0, spread * scale, 3))
                if v == 0:
                    anchor = anchor0
                dist = max(1., np.linalg.norm(anchor - tgt[0]))
                kwargs = dict(dz=.08 if tier == "high" else .25,
                              track_target=tier == 'high' and np.ptp(tgt, axis=0).max() > .01,
                              pitch_amp_deg=.5 if tier == 'high' else float(rng.uniform(0, 10)),
                              drift_scale=.002 if tier == 'high' else .1,
                              side_scale=.001 if tier == 'high' else .1,
                              look_noise_deg=.1 if tier == 'high' else 8.,
                              frac=.01 if tier == "high" else .12,
                              speed=.01 if tier == "high" else .08,
                              arc_deg=float(rng.uniform(4, 12)),
                              yaw_sweep_deg=float(rng.uniform(5, 15)),
                              pitch_deg=float(np.degrees(np.arctan2(tgt[0, 2] - anchor[2],
                                                    np.linalg.norm(tgt[0, :2] - anchor[:2])))),
                              dist=float(np.linalg.norm(anchor[:2] - tgt[0, :2])),
                              height=float(anchor[2] - tgt[0, 2]), side_deg=0,
                              lag_alpha=.9, turn_with_target=False)
                view = C.sample_view(rng, T, fps, intrinsics[v], anchor, dist, target=tgt,
                                     target_points=tgt, path_type=str(types[v]), strict=True,
                                     jitter=v == jitter_view, path_params=kwargs, max_tries=3,
                                     clearance_fn=clearance_fn, ray_fn=ray_fn, inside_fn=inside_fn,
                                     max_miss_frac=.65)
                views.append(view)
            score = preview_overlap(views, intrinsics, ray_fn)
            last = score
            accepted = score is not None and (
                (tier == "high" and score >= .65) or
                (tier == "medium" and .30 <= score < .55) or
                (tier == "low" and score < .22))
            if accepted:
                for v, view in enumerate(views):
                    view["meta"].update(view_id=v, overlap_target=tier, preview_overlap=score,
                                        rig_attempts=attempt + 1)
                return views
        except ValueError as e:
            last = str(e)
    raise RuntimeError(f"could not sample {tier} overlap rig: {last}")
