# Copyright (C) 2026. This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.
"""Multi-view camera paths for point-tracking data (p4d).

Every scene gets V synchronized cameras (default 4), each on an independently sampled path type:

    orbit     around a (possibly moving) target: radius, height, arc 30-360 deg, direction
    spin      in place: yaw sweep 30-150 deg with slow pitch sway
    crane     vertical rise/descent with tilt-to-target
    dolly     in/out along the line to the target
    follow    trails a moving object (offset, lag via exponential smoothing)
    handheld  smooth random walk (OU velocity), slowly wandering look direction
    static    near-static (sub-cm drift)

~25 % of views get jitter: Ornstein-Uhlenbeck noise on translation (sigma 1-4 cm) and rotation
(sigma 0.3-1.5 deg) plus a small high-frequency shake.

Validity (`validate_path`): clearance >= 0.3 m from geometry at every frame, target inside the frustum
for >= 80 % of frames (paths with a target), median depth (sampled rays) >= 0.3 m at every checked frame.
`sample_view` resamples on failure and falls back to `static` at the anchor.

This module is numpy-only (unit-testable without Blender); geometry queries are callbacks:
    clearance_fn(points [P,3]) -> distance to nearest geometry [P]
    ray_fn(origins [R,3], dirs [R,3]) -> hit distance [R] (inf = miss)
`blender_apply_path` keyframes a Blender object (imports bpy lazily).

Conventions: world Z up (Blender). Rotations R_cw [T,3,3] map camera -> world with *Blender camera axes*
(x right, y up, -z forward). OpenCV extrinsics: E = inv([R_cw @ diag(1,-1,-1) | t]).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

PATH_TYPES = ("orbit", "spin", "crane", "dolly", "follow", "handheld", "static")
DEFAULT_WEIGHTS = {"orbit": 0.2, "spin": 0.12, "crane": 0.14, "dolly": 0.14, "follow": 0.18, "handheld": 0.14,
                   "static": 0.08}
NEEDS_TARGET = {"orbit", "crane", "dolly", "follow"}


@dataclasses.dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_blender(cls, lens_mm, sensor_width_mm, width, height):
        fx = lens_mm / sensor_width_mm * width
        return cls(width, height, fx, fx, width / 2, height / 2)

    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


# ---------------------------------------------------------------------------------------------- geometry
def look_rotation(forward, up=(0.0, 0.0, 1.0), roll=0.0):
    """forward [...,3] -> R_cw [...,3,3] (Blender camera axes: columns x, y, -forward)."""
    f = np.asarray(forward, np.float64)
    f = f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-12)
    up = np.broadcast_to(np.asarray(up, np.float64), f.shape)
    x = np.cross(f, up)
    bad = np.linalg.norm(x, axis=-1) < 1e-6  # looking straight up/down: pick any horizontal x
    if np.any(bad):
        x = np.where(bad[..., None], np.cross(f, np.array([0.0, 1.0, 0.0])), x)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = np.cross(x, f)
    R = np.stack([x, y, -f], axis=-1)
    roll = np.broadcast_to(np.asarray(roll, np.float64), f.shape[:-1])
    if np.any(roll != 0):
        c, s = np.cos(roll), np.sin(roll)
        Rz = np.zeros(roll.shape + (3, 3))
        Rz[..., 0, 0], Rz[..., 0, 1], Rz[..., 1, 0], Rz[..., 1, 1], Rz[..., 2, 2] = c, -s, s, c, 1
        R = R @ Rz
    return R


def small_rotation(rotvec):
    """rotation vectors [...,3] (rad) -> rotation matrices (Rodrigues)."""
    r = np.asarray(rotvec, np.float64)
    th = np.linalg.norm(r, axis=-1, keepdims=True)
    k = r / np.maximum(th, 1e-12)
    K = np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def extrinsics_cv(pos, R_cw):
    """[T,3], [T,3,3] (Blender cam axes) -> OpenCV world->camera [T,4,4]."""
    T = len(pos)
    M = np.tile(np.eye(4), (T, 1, 1))
    M[:, :3, :3] = R_cw @ np.diag([1.0, -1.0, -1.0])
    M[:, :3, 3] = pos
    return np.linalg.inv(M)


def project(points, pos, R_cw, intr: Intrinsics):
    """points [T,P,3] or [P,3] (broadcast over T) -> uv [T,P,2], z [T,P] (camera depth)."""
    E = extrinsics_cv(pos, R_cw)
    pts = np.asarray(points, np.float64)
    if pts.ndim == 2:
        pts = np.broadcast_to(pts, (len(pos),) + pts.shape)
    cam = np.einsum("tij,tpj->tpi", E[:, :3, :3], pts) + E[:, None, :3, 3]
    z = cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam[..., 0] / z * intr.fx + intr.cx
        v = cam[..., 1] / z * intr.fy + intr.cy
    return np.stack([u, v], -1), z


def smoothstep(s):
    s = np.clip(s, 0, 1)
    return s * s * (3 - 2 * s)


def ema(x, alpha):
    """exponential moving average along axis 0 (lag for follow cameras)."""
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = out[t - 1] + alpha * (x[t] - out[t - 1])
    return out


def ou_process(rng, T, dim, sigma, theta, dt):
    """Stationary Ornstein-Uhlenbeck process [T,dim] with std sigma, mean reversion rate theta (1/s)."""
    x = np.zeros((T, dim))
    x[0] = rng.normal(0, sigma, dim)
    a = math.exp(-theta * dt)
    b = sigma * math.sqrt(1 - a * a)
    for t in range(1, T):
        x[t] = a * x[t - 1] + b * rng.normal(0, 1, dim)
    return x


# ---------------------------------------------------------------------------------------------- paths
def _target_at(target, T):
    if target is None:
        return None
    t = np.asarray(target, np.float64)
    return np.broadcast_to(t, (T, 3)).copy() if t.ndim == 1 else t


def path_orbit(rng, T, anchor, target, scale, **kw):
    tgt = _target_at(target, T)
    c0 = tgt[0]
    off = anchor - c0
    r0 = np.linalg.norm(off[:2])
    r = kw.get("radius") or float(np.clip(r0 if r0 > 0.3 * scale else rng.uniform(0.7, 1.3) * scale,
                                          0.5 * scale, 2.0 * scale))
    h = kw.get("height") if kw.get("height") is not None else float(
        np.clip(anchor[2] - c0[2], 0.15 * r, 0.9 * r) if anchor is not None else rng.uniform(0.2, 0.8) * r)
    arc = math.radians(kw.get("arc_deg") or rng.uniform(30, 360))
    direction = kw.get("direction") or rng.choice([-1, 1])
    az0 = math.atan2(off[1], off[0])
    s = smoothstep(np.linspace(0, 1, T)) if arc < math.pi else np.linspace(0, 1, T)
    az = az0 + direction * arc * s
    r_t = r * (1 + rng.uniform(-0.1, 0.1) * np.sin(np.pi * np.linspace(0, 1, T)))
    pos = np.stack([tgt[:, 0] + r_t * np.cos(az), tgt[:, 1] + r_t * np.sin(az), tgt[:, 2] + h], -1)
    look = tgt - pos
    return pos, look_rotation(look), dict(radius=r, height=h, arc_deg=math.degrees(arc), direction=int(direction))


def path_spin(rng, T, anchor, target, scale, **kw):
    yaw_sweep = math.radians(kw.get("yaw_sweep_deg") or rng.uniform(30, 150)) * rng.choice([-1, 1])
    pitch0 = math.radians(kw.get("pitch_deg") if kw.get("pitch_deg") is not None else rng.uniform(-20, 5))
    pitch_amp = math.radians(rng.uniform(0, 10))
    if target is not None:
        d = _target_at(target, T)[0] - anchor
        yaw0 = math.atan2(d[1], d[0]) - yaw_sweep / 2
    else:
        yaw0 = rng.uniform(-math.pi, math.pi)
    s = smoothstep(np.linspace(0, 1, T))
    yaw = yaw0 + yaw_sweep * s
    pitch = pitch0 + pitch_amp * np.sin(2 * np.pi * np.linspace(0, 1, T) * rng.uniform(0.5, 1.5))
    f = np.stack([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)], -1)
    pos = np.broadcast_to(anchor, (T, 3)).copy()
    return pos, look_rotation(f), dict(yaw_sweep_deg=math.degrees(yaw_sweep), pitch_deg=math.degrees(pitch0),
                                       pitch_amp_deg=math.degrees(pitch_amp))


def path_crane(rng, T, anchor, target, scale, **kw):
    tgt = _target_at(target, T)
    dz = kw.get("dz") or float(rng.uniform(0.25, 0.7) * scale * rng.choice([-1, 1]))
    drift = rng.normal(0, 0.1 * scale, 2)
    s = smoothstep(np.linspace(0, 1, T))
    pos = np.broadcast_to(anchor, (T, 3)).copy()
    pos[:, 2] += dz * s
    pos[:, :2] += np.outer(s, drift)
    return pos, look_rotation(tgt - pos), dict(dz=dz, drift=drift.tolist())


def path_dolly(rng, T, anchor, target, scale, **kw):
    tgt = _target_at(target, T)
    d0 = tgt[0] - anchor
    dist = np.linalg.norm(d0)
    frac = kw.get("frac") or float(rng.uniform(0.2, 0.5) * rng.choice([-1, 1]))  # + in, - out
    s = smoothstep(np.linspace(0, 1, T))
    u = d0 / max(dist, 1e-9)
    side = np.cross(u, [0, 0, 1.0])
    side_amt = rng.normal(0, 0.1) * dist
    pos = anchor + np.outer(s * frac * dist, u) + np.outer(s * side_amt, side)
    return pos, look_rotation(tgt - pos), dict(frac=frac, dist0=float(dist), side=float(side_amt))


def path_follow(rng, T, anchor, target, scale, **kw):
    tgt = _target_at(target, T)
    lag = kw.get("lag_alpha") or float(rng.uniform(0.08, 0.3))
    sm = ema(tgt, lag)
    vel = ema(np.gradient(sm, axis=0), 0.2)
    off0 = anchor - tgt[0]
    # heading = direction of travel; hold the last heading while the target is (nearly) still, starting from the
    # anchor's side, so the camera never flips around a stopping / tumbling object
    heading = np.zeros((T, 2))
    h = -off0[:2] / max(np.linalg.norm(off0[:2]), 1e-6)
    min_speed = 0.02 * scale / fps if (fps := kw.get("fps")) else 1e-3
    for t in range(T):
        sp = np.linalg.norm(vel[t, :2])
        if sp > min_speed:
            h = 0.85 * h + 0.15 * vel[t, :2] / sp
            h /= max(np.linalg.norm(h), 1e-9)
        heading[t] = h
    dist = kw.get("dist") or float(np.clip(np.linalg.norm(off0[:2]), 0.6 * scale, 1.6 * scale))
    side_ang = math.radians(kw.get("side_deg") if kw.get("side_deg") is not None else rng.uniform(-60, 60))
    c, s_ = math.cos(math.pi + side_ang), math.sin(math.pi + side_ang)
    back = np.stack([heading[:, 0] * c - heading[:, 1] * s_, heading[:, 0] * s_ + heading[:, 1] * c], -1)
    back = ema(back, 0.15)
    back /= np.maximum(np.linalg.norm(back, axis=1, keepdims=True), 1e-6)
    h = kw.get("height") if kw.get("height") is not None else float(np.clip(anchor[2] - tgt[0, 2], 0.2 * dist, dist))
    pos = np.concatenate([sm[:, :2] + dist * back, sm[:, 2:3] + h], 1)
    return pos, look_rotation(sm - pos), dict(lag_alpha=lag, dist=dist, side_deg=math.degrees(side_ang), height=h)


def path_handheld(rng, T, anchor, target, scale, fps=24.0, **kw):
    dt = 1.0 / fps
    speed = kw.get("speed") or float(rng.uniform(0.15, 0.5) * scale / 3)
    vel = ou_process(rng, T, 3, speed, theta=0.8, dt=dt)
    vel[:, 2] *= 0.3
    pos = anchor + np.cumsum(vel * dt, axis=0)
    if target is not None:
        look = _target_at(target, T) - pos
        yaw_n = ou_process(rng, T, 2, math.radians(8), theta=0.5, dt=dt)
        base = look_rotation(look)
        R = base @ small_rotation(np.concatenate([yaw_n, np.zeros((T, 1))], 1))
    else:
        yaw = rng.uniform(-math.pi, math.pi) + np.cumsum(ou_process(rng, T, 1, math.radians(15), 0.5, dt)[:, 0] * dt)
        pitch = np.full(T, math.radians(rng.uniform(-15, 0)))
        f = np.stack([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)], -1)
        R = look_rotation(f)
    return pos, R, dict(speed=speed)


def path_static(rng, T, anchor, target, scale, fps=24.0, **kw):
    drift = ou_process(rng, T, 3, 0.003, theta=0.3, dt=1.0 / fps)
    pos = anchor + drift
    if target is not None:
        look = np.broadcast_to(_target_at(target, T)[0] - anchor, (T, 3))
    else:
        yaw = rng.uniform(-math.pi, math.pi)
        look = np.broadcast_to([math.cos(yaw), math.sin(yaw), -0.1], (T, 3))
    return pos, look_rotation(look), dict(drift_sigma_m=0.003)


PATHS = {"orbit": path_orbit, "spin": path_spin, "crane": path_crane, "dolly": path_dolly, "follow": path_follow,
         "handheld": path_handheld, "static": path_static}


def apply_jitter(rng, pos, R, fps, trans_sigma=None, rot_sigma_deg=None):
    """OU translation + rotation noise with a small high-frequency shake (camera frame rotation)."""
    T = len(pos)
    dt = 1.0 / fps
    ts = trans_sigma if trans_sigma is not None else float(rng.uniform(0.01, 0.04))
    rs = math.radians(rot_sigma_deg if rot_sigma_deg is not None else float(rng.uniform(0.3, 1.5)))
    theta = float(rng.uniform(1.0, 4.0))
    dpos = ou_process(rng, T, 3, ts, theta, dt) + ou_process(rng, T, 3, 0.15 * ts, 25.0, dt)
    drot = ou_process(rng, T, 3, rs, theta, dt) + ou_process(rng, T, 3, 0.15 * rs, 25.0, dt)
    params = dict(trans_sigma_m=ts, rot_sigma_deg=math.degrees(rs), theta_per_s=theta, shake_frac=0.15,
                  shake_theta_per_s=25.0)
    return pos + dpos, R @ small_rotation(drot), params, dpos, drot


# ---------------------------------------------------------------------------------------------- validity
def validate_path(pos, R, intr: Intrinsics, clearance_fn=None, ray_fn=None, target=None, min_clearance=0.3,
                  min_target_frac=0.8, min_median_depth=0.3, check_every=4, grid=(8, 6), margin_px=0, inside_fn=None,
                  max_miss_frac=None):
    """-> (ok, report). target: [T,3] or [T,P,3] points (all must be in frame to count).
    inside_fn(points [T,3]) -> bool [T]: optional region constraint (e.g. inside the room)."""
    T = len(pos)
    rep = {}
    if inside_fn is not None:
        ins = np.asarray(inside_fn(pos), bool)
        rep["inside_frac"] = float(ins.mean())
        if not ins.all():
            return False, dict(rep, reason="outside_region")
    if clearance_fn is not None:
        cl = np.asarray(clearance_fn(pos))
        rep["min_clearance_m"] = float(cl.min())
        if cl.min() < min_clearance:
            return False, dict(rep, reason="clearance")
    if target is not None:
        tp = np.asarray(target, np.float64)
        if tp.ndim == 1:  # fixed point
            tp = np.broadcast_to(tp, (T, 1, 3))
        elif tp.ndim == 2:  # [T,3] moving point
            tp = tp[:, None]
        uv, z = project(tp, pos, R, intr)
        inside = (z > 0.05) & (uv[..., 0] >= margin_px) & (uv[..., 0] < intr.width - margin_px) & \
                 (uv[..., 1] >= margin_px) & (uv[..., 1] < intr.height - margin_px)
        frac = float(inside.all(-1).mean())
        rep["target_in_frame_frac"] = frac
        if frac < min_target_frac:
            return False, dict(rep, reason="target_frustum")
    if ray_fn is not None:
        gx, gy = grid
        us = (np.arange(gx) + 0.5) / gx * intr.width
        vs = (np.arange(gy) + 0.5) / gy * intr.height
        uu, vv = np.meshgrid(us, vs)
        d_cam = np.stack([(uu.ravel() - intr.cx) / intr.fx, (vv.ravel() - intr.cy) / intr.fy,
                          np.ones(uu.size)], 1)  # OpenCV camera frame, z = 1
        worst = np.inf
        for t in range(0, T, check_every):
            Rcv = R[t] @ np.diag([1.0, -1.0, -1.0])
            dirs = d_cam @ Rcv.T
            norms = np.linalg.norm(dirs, axis=1)
            dist = np.asarray(ray_fn(np.broadcast_to(pos[t], dirs.shape), dirs / norms[:, None]))
            zdep = dist / norms  # ray length -> camera z
            med = float(np.median(zdep))
            worst = min(worst, med)
            miss = float(np.mean(~np.isfinite(zdep)))
            rep["max_miss_frac"] = max(rep.get("max_miss_frac", 0.0), miss)
            if max_miss_frac is not None and miss > max_miss_frac:
                return False, dict(rep, reason="sky", frame=t)
            if med < min_median_depth:
                rep["median_depth_min_m"] = med
                return False, dict(rep, reason="median_depth", frame=t)
        rep["median_depth_min_m"] = float(worst)
    return True, rep


def sample_view(rng, T, fps, intr, anchor, scale, target=None, target_points=None, path_type=None, weights=None,
                jitter_prob=0.25, clearance_fn=None, ray_fn=None, max_tries=12, **val_kw):
    """Sample one valid view. target: [T,3] (look target / followed object centre) or None.
    target_points: [T,P,3] points that must stay in frame (defaults to target). -> dict(pos, R, meta)."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    if target is None:
        for k in NEEDS_TARGET:
            weights.pop(k, None)
    types = list(weights)
    p = np.array([weights[k] for k in types], np.float64)
    p /= p.sum()
    tried = []
    for attempt in range(max_tries):
        pt = path_type if (path_type and attempt < max_tries // 2) else str(rng.choice(types, p=p))
        if pt in NEEDS_TARGET and target is None:
            pt = "handheld"
        fn = PATHS[pt]
        kw = dict(fps=fps) if pt in ("handheld", "static", "follow") else {}
        pos, R, params = fn(rng, T, np.asarray(anchor, np.float64), target, scale, **kw)
        jit = bool(rng.random() < jitter_prob)
        jparams = None
        if jit:
            pos, R, jparams, _, _ = apply_jitter(rng, pos, R, fps)
        tp = target_points if target_points is not None else (target if pt in NEEDS_TARGET else None)
        ok, rep = validate_path(pos, R, intr, clearance_fn=clearance_fn, ray_fn=ray_fn,
                                target=tp if pt in NEEDS_TARGET else None, **val_kw)
        tried.append(dict(path_type=pt, jitter=jit, **rep))
        if ok:
            return dict(pos=pos, R=R, meta=dict(path_type=pt, params=_jsonable(params), jitter=jit,
                                                jitter_params=_jsonable(jparams), attempts=attempt + 1,
                                                validation=_jsonable(rep), rejected=tried[:-1]))
    pos, R, params = path_static(rng, T, np.asarray(anchor, np.float64), target, scale, fps=fps)
    ok, rep = validate_path(pos, R, intr, clearance_fn=clearance_fn, ray_fn=ray_fn, **val_kw)
    return dict(pos=pos, R=R, meta=dict(path_type="static", params=_jsonable(params), jitter=False, jitter_params=None,
                                        attempts=max_tries + 1, validation=_jsonable(rep), fallback=True,
                                        fallback_valid=bool(ok), rejected=tried))


def assign_path_types(rng, n_views, has_target, weights=None):
    """Distinct path types across views where possible."""
    w = dict(weights or DEFAULT_WEIGHTS)
    if not has_target:
        for k in NEEDS_TARGET:
            w.pop(k, None)
    types = list(w)
    p = np.array([w[k] for k in types])
    p = p / p.sum()
    k = min(n_views, len(types))
    chosen = list(rng.choice(types, size=k, replace=False, p=p))
    while len(chosen) < n_views:
        chosen.append(str(rng.choice(types, p=p)))
    return [str(c) for c in chosen]


def _jsonable(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


# ---------------------------------------------------------------------------------------------- Blender glue
def blender_apply_path(obj, pos, R, frame_start=1, clear_parent=True):
    """Keyframe obj (camera or camera rig) at every frame with the given world pose (Blender camera axes)."""
    import bpy  # noqa: F401
    from mathutils import Matrix

    if clear_parent and obj.parent is not None:
        mw = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = mw
    obj.animation_data_clear()
    for c in list(obj.constraints):
        obj.constraints.remove(c)
    obj.rotation_mode = "XYZ"
    prev = None
    for t in range(len(pos)):
        M = Matrix.Identity(4)
        for i in range(3):
            for j in range(3):
                M[i][j] = float(R[t, i, j])
            M[i][3] = float(pos[t, i])
        obj.matrix_world = M
        if prev is not None:
            obj.rotation_euler.make_compatible(prev)
        prev = obj.rotation_euler.copy()
        obj.keyframe_insert("location", frame=frame_start + t)
        obj.keyframe_insert("rotation_euler", frame=frame_start + t)
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


def blender_bvh_callbacks(objects=None, depsgraph=None):
    """BVH over evaluated meshes -> (clearance_fn, ray_fn) for validate_path."""
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    objs = objects if objects is not None else [o for o in bpy.context.scene.objects
                                                 if o.type == "MESH" and not o.hide_render]
    verts, polys = [], []
    off = 0
    for o in objs:
        eo = o.evaluated_get(dg)
        try:
            me = eo.to_mesh()
        except RuntimeError:
            continue
        if me is None or len(me.vertices) == 0:
            eo.to_mesh_clear()
            continue
        M = np.array(eo.matrix_world)
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        me.calc_loop_triangles()
        tri = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
        verts.append(co)
        polys.append(tri.reshape(-1, 3) + off)
        off += len(co)
        eo.to_mesh_clear()
    V = np.concatenate(verts) if verts else np.zeros((0, 3))
    F = np.concatenate(polys) if polys else np.zeros((0, 3), np.int64)
    bvh = BVHTree.FromPolygons([tuple(v) for v in V], [tuple(f) for f in F.tolist()], all_triangles=True)

    def clearance_fn(points):
        out = []
        for p in np.asarray(points):
            hit = bvh.find_nearest(Vector(p))
            out.append(np.inf if hit[0] is None else hit[3])
        return np.array(out)

    def ray_fn(origins, dirs):
        out = []
        for o, d in zip(np.asarray(origins), np.asarray(dirs)):
            hit = bvh.ray_cast(Vector(o), Vector(d))
            out.append(np.inf if hit[0] is None else hit[3])
        return np.array(out)

    return clearance_fn, ray_fn, bvh
