# Copyright (C) 2026. This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.
"""Articulated Infinigen-Sim assets (drawer, cabinet, door, dishwasher, ...) with animated joints in rooms.

Infinigen-Sim assets are one mesh object driven by a geometry-nodes tree containing hinge / sliding joint node
groups whose "Value" socket sets the joint state; Infinigen itself only moves them in the MuJoCo visualiser.
Here every unlinked joint Value is keyframed with a sampled profile:

    open             min -> a
    close            a -> min
    partial          b -> c (random sub-range)
    open_close       min -> a -> min
    repeated         min <-> a, 1.5-2.5 cycles

a = min + U(0.6, 0.9) * (max - min) (never fully to the limit: avoids self-intersection). Unlimited joints
(min = max = 0) get 1.3 rad (hinge) / 0.3 m (slide). Joints start at a random frame and last 40-100 % of the
clip; per-joint profiles are independent. Assets are placed on the floor against a wall with their front (mean
joint motion direction) facing into the room, rejecting spots whose AABB overlaps existing furniture.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

PROFILES = ("open", "close", "partial", "open_close", "repeated")
# (asset, weight): floor-standing, human-scale assets first (small countertop ones are hard to see on the floor)
DEFAULT_ASSETS = {"cabinet": 3, "drawer": 3, "dishwasher": 2, "refrigerator": 2, "door": 1, "oven": 1,
                  "microwave": 0.5, "toaster": 0.3}


def _walk(tree, seen, out):
    if tree is None or tree.name in seen:
        return
    seen.add(tree.name)
    for n in tree.nodes:
        if n.type == "GROUP" and n.node_tree is not None:
            nm = n.node_tree.name.lower().replace(" ", "").replace("_", "")
            if ("hingejoint" in nm or "slidingjoint" in nm) and "Value" in n.inputs:
                out.append((tree, n))
            _walk(n.node_tree, seen, out)


class _Driver:
    """Where a joint's Value really comes from: the socket itself, an upstream Value node, or a modifier input."""

    def __init__(self, kind, target, key=None):
        self.kind, self.target, self.key = kind, target, key

    def set(self, v, frame):
        if self.kind == "socket":
            self.target.default_value = v
            self.target.keyframe_insert("default_value", frame=frame)
        elif self.kind == "value_node":
            self.target.outputs[0].default_value = v
            self.target.outputs[0].keyframe_insert("default_value", frame=frame)
        else:  # modifier input
            self.target[self.key] = v
            self.target.keyframe_insert(f'["{self.key}"]', frame=frame)


def _resolve(tree, sock, mod, top_tree, depth=0):
    """Follow a linked input upstream through reroutes to something we can animate."""
    if not sock.is_linked:
        return _Driver("socket", sock)
    if depth > 16:
        return None
    link = sock.links[0]
    node = link.from_node
    if node.type == "REROUTE":
        return _resolve(tree, node.inputs[0], mod, top_tree, depth + 1)
    if node.type == "VALUE":
        return _Driver("value_node", node)
    if node.type == "GROUP_INPUT" and tree is top_tree:
        name = link.from_socket.name
        for item in tree.interface.items_tree:
            if item.in_out == "INPUT" and item.name == name:
                return _Driver("modifier", mod, item.identifier)
    return None


def find_joints(obj):
    """-> [(tree, joint node, driver)] for every joint whose Value can be animated."""
    out = []
    for m in obj.modifiers:
        if m.type != "NODES":
            continue
        found = []
        _walk(m.node_group, set(), found)
        seen = set()
        for t, n in found:
            d = _resolve(t, n.inputs["Value"], m, m.node_group)
            if d is None:
                continue
            key = (d.kind, id(d.target), d.key) if d.kind != "socket" else ("socket", t.name, n.name)
            if key in seen:  # several joints fed by one value (e.g. duplicated doors) animate together
                continue
            seen.add(key)
            out.append((t, n, d))
    return out


def profile_curve(rng, profile, T):
    """-> s[T] in [0,1] (fraction of the joint's opening)."""
    t = np.arange(T)
    start = int(rng.integers(0, max(1, T // 3)))
    dur = int(max(8, rng.uniform(0.4, 1.0) * (T - start)))
    u = np.clip((t - start) / max(dur - 1, 1), 0, 1)
    ease = u * u * (3 - 2 * u)
    if profile == "open":
        return ease
    if profile == "close":
        return 1 - ease
    if profile == "partial":
        a, b = sorted(rng.uniform(0.1, 0.9, 2))
        return a + (b - a) * ease
    if profile == "open_close":
        return np.sin(np.pi * u) ** 2 * (u > 0)
    if profile == "repeated":
        cyc = rng.uniform(1.5, 2.5)
        return 0.5 - 0.5 * np.cos(2 * np.pi * cyc * u)
    raise ValueError(profile)


def animate_joints(rng, obj, frame_start, T, profiles=None):
    joints = []
    for tree, n, drv in find_joints(obj):
        kind = "hinge" if "hinge" in n.node_tree.name.lower() else "slide"
        lo = n.inputs["Min"].default_value if "Min" in n.inputs and not n.inputs["Min"].is_linked else 0.0
        hi = n.inputs["Max"].default_value if "Max" in n.inputs and not n.inputs["Max"].is_linked else 0.0
        if lo == 0.0 and hi == 0.0:
            hi = 1.3 if kind == "hinge" else 0.3
        amax = float(lo + rng.uniform(0.6, 0.9) * (hi - lo))
        prof = str(rng.choice(profiles or PROFILES))
        s = profile_curve(rng, prof, T)
        for t in range(T):
            drv.set(float(lo + (amax - lo) * s[t]), frame_start + t)
        label = n.inputs["Joint Label"].default_value if "Joint Label" in n.inputs and \
            not n.inputs["Joint Label"].is_linked else ""
        joints.append(dict(tree=tree.name, node=n.name, kind=kind, label=label, driver=drv.kind, min=float(lo),
                           max=float(hi),
                           amax=amax, profile=prof, s_range=[float(s.min()), float(s.max())]))
    return joints


def front_direction(obj, frame_a, frame_b):
    """Mean horizontal displacement of the vertices the joints move between two frames (object front)."""
    import bpy

    s = bpy.context.scene
    snaps = []
    for f in (frame_a, frame_b):
        s.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        eo = obj.evaluated_get(dg)
        me = eo.to_mesh()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        M = np.array(eo.matrix_world)
        snaps.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        eo.to_mesh_clear()
    if len(snaps[0]) != len(snaps[1]):
        return None
    d = snaps[1] - snaps[0]
    mv = np.linalg.norm(d, axis=1) > 1e-4
    if not mv.any():
        return None
    f = d[mv].mean(0)
    f[2] = 0
    n = np.linalg.norm(f)
    return f / n if n > 1e-6 else None


def spawn_asset(name, seed):
    from infinigen.core.sim import sim_factory as sf

    obj = sf.spawn_simready(name=name, seed=seed, export=False)
    if isinstance(obj, (list, tuple)):
        obj = obj[0]
    obj.name = f"sim_{name}_{seed}"
    return obj


def _overlaps(lo, hi, boxes, tol=0.01):
    return any(np.all(lo < b_hi - tol) and np.all(hi > b_lo + tol) for b_lo, b_hi in boxes)


def _motion_box(obj, frame_start, T, n=3):
    import bpy

    from infinigen.p4d.motion.objects import world_bbox

    s = bpy.context.scene
    lo, hi = world_bbox(obj)
    for f in np.linspace(frame_start, frame_start + T - 1, n).astype(int):
        s.frame_set(int(f))
        a, b = world_bbox(obj)
        lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    s.frame_set(frame_start)
    return lo, hi


def place_against_wall(rng, obj, room_bbox, occupied, frame_start, T, max_tries=40, margin=0.05):
    """Rotate so the front faces into the room; put it on the floor against a random wall (or, failing that, on a
    free floor spot facing the room centre); reject spots whose full-motion AABB overlaps existing furniture."""
    import bpy
    from mathutils import Vector

    from infinigen.p4d.motion.objects import world_bbox

    lo, hi = np.asarray(room_bbox[0], float), np.asarray(room_bbox[1], float)
    fr = front_direction(obj, frame_start, frame_start + T - 1) if T > 1 else None
    s = bpy.context.scene
    s.frame_set(frame_start)
    front_yaw = math.atan2(fr[1], fr[0]) if fr is not None else -math.pi / 2
    walls = [(0, lo[0], 0.0), (0, hi[0], math.pi), (1, lo[1], math.pi / 2), (1, hi[1], -math.pi / 2)]
    reasons = {"no_span": 0, "static_overlap": 0, "motion_overlap": 0}
    wall_only = "door" in obj.name.lower() and "dishwasher" not in obj.name.lower()  # a free-standing door frame
    for attempt in range(max_tries):
        free_floor = attempt >= max_tries // 2 and not wall_only
        if free_floor:
            x, y = rng.uniform(lo[0] + 0.5, hi[0] - 0.5), rng.uniform(lo[1] + 0.5, hi[1] - 0.5)
            ctr_room = (lo + hi) / 2
            inward_yaw = math.atan2(ctr_room[1] - y, ctr_room[0] - x)
        else:
            axis, coord, inward_yaw = walls[int(rng.integers(0, 4))]
        obj.rotation_euler.z += inward_yaw - front_yaw
        front_yaw = inward_yaw
        bpy.context.view_layer.update()
        mn, mx = world_bbox(obj)
        size = mx - mn
        if free_floor:
            ctr = np.array([x, y, 0.0])
        else:
            other = 1 - axis
            span_lo, span_hi = lo[other] + size[other] / 2 + 0.1, hi[other] - size[other] / 2 - 0.1
            if span_hi <= span_lo:
                reasons["no_span"] += 1
                continue
            ctr = np.zeros(3)
            ctr[other] = rng.uniform(span_lo, span_hi)
            ctr[axis] = coord + (size[axis] / 2 + margin) * (1 if coord == lo[axis] else -1)
        cur = (mn + mx) / 2
        obj.location += Vector((ctr[0] - cur[0], ctr[1] - cur[1], lo[2] - mn[2] + 0.001))
        bpy.context.view_layer.update()
        mn, mx = world_bbox(obj)
        # cheap reject: static box grown 0.3 m towards the front (doors/drawers swing out)
        f = np.array([math.cos(front_yaw), math.sin(front_yaw), 0.0])
        g_lo, g_hi = np.minimum(mn, mn + 0.3 * f), np.maximum(mx, mx + 0.3 * f)
        if _overlaps(g_lo, g_hi, occupied):
            reasons["static_overlap"] += 1
            continue
        box_lo, box_hi = _motion_box(obj, frame_start, T)
        if _overlaps(box_lo, box_hi, occupied) or np.any(box_lo[:2] < lo[:2] - 0.02) or np.any(box_hi[:2] > hi[:2] + 0.02):
            reasons["motion_overlap"] += 1
            continue
        occupied.append((box_lo, box_hi))
        return dict(mode="free_floor" if free_floor else "wall", center=ctr.tolist(), size=size.tolist(),
                    attempts=attempt + 1)
    logger.info(f"placement failed for {obj.name}: {reasons}")
    return None


def add_articulated(rng, room_bbox, occupied, frame_start, T, n=(2, 3), assets=DEFAULT_ASSETS, profiles=None):
    """Spawn + animate + place n articulated assets. Returns (objects, summary)."""
    import bpy

    k = int(rng.integers(n[0], n[1] + 1))
    pool = dict(assets) if isinstance(assets, dict) else {a: 1.0 for a in assets}
    keys = list(pool)
    p = np.array([pool[x] for x in keys], float)
    names = list(rng.choice(keys, size=k, replace=False, p=p / p.sum()))
    objs, summary = [], []
    for nm in names:
        seed = int(rng.integers(0, 10**6))
        try:
            obj = spawn_asset(nm, seed)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"spawn {nm} failed: {e!r}")
            summary.append(dict(asset=nm, seed=seed, error=repr(e)[:200]))
            continue
        joints = animate_joints(rng, obj, frame_start, T, profiles=profiles)
        if not joints:
            logger.warning(f"{nm}: no animatable joints; removed")
            bpy.data.objects.remove(obj, do_unlink=True)
            summary.append(dict(asset=nm, seed=seed, error="no joints"))
            continue
        place = place_against_wall(rng, obj, room_bbox, occupied, frame_start, T)
        if place is None:
            bpy.data.objects.remove(obj, do_unlink=True)
            summary.append(dict(asset=nm, seed=seed, error="no free wall spot"))
            continue
        obj["p4d_kind"] = "articulated"
        obj["p4d_class"] = f"articulated_{nm}"
        objs.append(obj)
        summary.append(dict(asset=nm, seed=seed, object=obj.name, joints=joints, placement=place))
        logger.info(f"p4d articulated {obj.name}: {len(joints)} joints {[j['profile'] for j in joints]}")
    return objs, summary
