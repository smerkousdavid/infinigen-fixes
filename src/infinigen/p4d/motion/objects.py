# Copyright (C) 2026. This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.
"""Physically plausible indoor object motion (replaces floating random walks).

* rigid-body drops / tosses of small objects onto furniture and the floor (Blender Bullet rigid bodies,
  furniture + room shell passive MESH colliders, actives CONVEX_HULL)
* slides along support surfaces and rolling balls: rigid bodies launched with an initial velocity - the
  standard "kinematic hand-off" trick: the object is animated (kinematic) for the first two frames, then handed
  to the simulation, which inherits the velocity implied by those keyframes; friction/rolling come from Bullet
* pushed chairs: keyframed translation away from the table (+ small yaw) with ease-in/out, snapped to the floor

All moved objects get obj["p4d_kind"] = "rigid" and obj["p4d_motion"] = kind of motion. After
`bake()`, frames are read from the point cache, so evaluated transforms are the simulated ones.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def world_bbox(o):
    import bpy

    dg = bpy.context.evaluated_depsgraph_get()
    eo = o.evaluated_get(dg)
    M = np.array(eo.matrix_world)
    bb = np.array([list(v) for v in eo.bound_box])
    w = bb @ M[:3, :3].T + M[:3, 3]
    return w.min(0), w.max(0)


def _ensure_world(frame_start, frame_end, substeps=20):
    import bpy

    s = bpy.context.scene
    if s.rigidbody_world is None:
        with bpy.context.temp_override(scene=s):
            bpy.ops.rigidbody.world_add()
    rw = s.rigidbody_world
    if rw.collection is None:
        rw.collection = bpy.data.collections.new("RigidBodyWorld")
    rw.substeps_per_frame = substeps
    rw.solver_iterations = 20
    rw.point_cache.frame_start = frame_start
    rw.point_cache.frame_end = frame_end
    return rw


def _add_rb(o, kind, shape, mass=1.0, friction=0.6, restitution=0.2):
    import bpy

    s = bpy.context.scene
    if o.name not in s.rigidbody_world.collection.objects:
        s.rigidbody_world.collection.objects.link(o)
    rb = o.rigid_body
    rb.type = kind
    rb.collision_shape = shape
    rb.mass = mass
    rb.friction = friction
    rb.restitution = restitution
    rb.use_margin = True
    rb.collision_margin = 0.002
    rb.linear_damping = 0.05
    rb.angular_damping = 0.1
    return rb


def _launch(o, frame_start, v, w=(0.0, 0.0, 0.0), fps=24.0):
    """Kinematic for 2 frames, moving with linear velocity v (m/s) and angular velocity w (rad/s), then dynamic."""
    rb = o.rigid_body
    dt = 1.0 / fps
    p0, r0 = o.location.copy(), o.rotation_euler.copy()
    rb.kinematic = True
    o.keyframe_insert("location", frame=frame_start)
    o.keyframe_insert("rotation_euler", frame=frame_start)
    rb.keyframe_insert("kinematic", frame=frame_start)
    o.location = p0 + type(p0)(v) * dt
    o.rotation_euler = (r0.x + w[0] * dt, r0.y + w[1] * dt, r0.z + w[2] * dt)
    o.keyframe_insert("location", frame=frame_start + 1)
    o.keyframe_insert("rotation_euler", frame=frame_start + 1)
    rb.kinematic = True
    rb.keyframe_insert("kinematic", frame=frame_start + 1)
    rb.kinematic = False
    rb.keyframe_insert("kinematic", frame=frame_start + 2)
    o.location, o.rotation_euler = p0, r0


def _support_height(bvh, x, y, z_from):
    from mathutils import Vector

    loc, nrm, _, _ = bvh.ray_cast(Vector((x, y, z_from)), Vector((0, 0, -1)))
    return (None, None) if loc is None else (loc.z, nrm)


def add_ball(radius, name, material=None):
    import bpy

    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=32, ring_count=16)
    o = bpy.context.active_object
    o.name = name
    bpy.ops.object.shade_smooth()
    mat = material or bpy.data.materials.new(name + "_mat")
    if material is None:
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        chk = mat.node_tree.nodes.new("ShaderNodeTexChecker")  # texture so rolling is visible
        chk.inputs["Scale"].default_value = 6.0
        c = np.random.default_rng(abs(hash(name)) % 2**31).uniform(0.1, 0.9, 3)
        chk.inputs["Color1"].default_value = (*c, 1)
        chk.inputs["Color2"].default_value = (*(1 - c), 1)
        mat.node_tree.links.new(chk.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.5
    o.data.materials.append(mat)
    return o


def setup_physics(rng, scene_objects, small_objects, room_bbox, frame_start, frame_end, fps=24.0, n_drop=(3, 6),
                  n_slide=(1, 2), n_roll=(1, 3), chairs=(), push_chairs=(1, 2), bvh=None):
    """Configure the rigid-body world and motion. Returns a summary dict (per-object motion type + params)."""
    import bpy
    from mathutils import Vector

    rw = _ensure_world(frame_start, frame_end)
    lo, hi = np.asarray(room_bbox[0], float), np.asarray(room_bbox[1], float)
    movers = set()
    summary = {"drops": [], "slides": [], "rolls": [], "chairs": []}
    small = [o for o in small_objects if o.type == "MESH"]
    rng.shuffle(small)
    # passive colliders: everything else that is a mesh (room shell + furniture)
    nd = int(rng.integers(n_drop[0], n_drop[1] + 1))
    ns = int(rng.integers(n_slide[0], n_slide[1] + 1))
    drops, slides = small[:nd], small[nd:nd + ns]
    chosen = set(o.name for o in drops + slides)
    for o in scene_objects:
        if o.type != "MESH" or o.name in chosen or o in chairs:
            continue
        try:
            _add_rb(o, "PASSIVE", "MESH", friction=0.7, restitution=0.1)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"passive rb {o.name}: {e}")
    for o in drops:  # lift 0.3-1.2 m above its support and toss with a small velocity + spin
        mn, mx = world_bbox(o)
        size = float(np.max(mx - mn))
        if size > 0.8:
            continue
        mass = float(np.clip(300 * np.prod(np.maximum(mx - mn, 0.02)), 0.05, 5.0))
        _add_rb(o, "ACTIVE", "CONVEX_HULL", mass=mass, friction=0.6, restitution=float(rng.uniform(0.05, 0.35)))
        h = float(rng.uniform(0.3, 1.2))
        o.location.z += h
        v = (float(rng.normal(0, 0.6)), float(rng.normal(0, 0.6)), float(rng.uniform(-0.5, 1.0)))
        w = tuple(float(x) for x in rng.normal(0, 3.0, 3))
        _launch(o, frame_start, v, w, fps)
        o["p4d_kind"], o["p4d_motion"] = "rigid", "drop"
        movers.add(o.name)
        summary["drops"].append(dict(object=o.name, lift_m=h, v=v, w=w, mass=mass))
    for o in slides:  # push along its support surface (it will slide / tip / fall off edges)
        mn, mx = world_bbox(o)
        if np.max(mx - mn) > 0.8:
            continue
        _add_rb(o, "ACTIVE", "CONVEX_HULL", mass=0.5, friction=float(rng.uniform(0.2, 0.45)), restitution=0.05)
        ang = rng.uniform(0, 2 * math.pi)
        spd = float(rng.uniform(0.6, 1.6))
        v = (spd * math.cos(ang), spd * math.sin(ang), 0.0)
        o.location.z += 0.003
        _launch(o, frame_start, v, (0.0, 0.0, float(rng.normal(0, 1.0))), fps)
        o["p4d_kind"], o["p4d_motion"] = "rigid", "slide"
        movers.add(o.name)
        summary["slides"].append(dict(object=o.name, speed_mps=spd, heading_deg=math.degrees(ang)))
    nr = int(rng.integers(n_roll[0], n_roll[1] + 1))
    for i in range(nr):  # balls rolling across the floor
        r = float(rng.uniform(0.06, 0.16))
        b = add_ball(r, f"p4d_ball_{i}")
        for _ in range(40):
            x, y = rng.uniform(lo[0] + 0.6, hi[0] - 0.6), rng.uniform(lo[1] + 0.6, hi[1] - 0.6)
            zf = lo[2]
            if bvh is not None:
                zh, _ = _support_height(bvh, x, y, hi[2] - 0.05)
                if zh is None or zh > lo[2] + 0.05:  # need bare floor here
                    continue
                zf = zh
            break
        b.location = (x, y, zf + r + 0.002)
        _add_rb(b, "ACTIVE", "SPHERE", mass=float(4 / 3 * math.pi * r**3 * 400), friction=0.5, restitution=0.4)
        b.rigid_body.linear_damping, b.rigid_body.angular_damping = 0.02, 0.05
        ctr = (lo + hi) / 2
        ang = math.atan2(ctr[1] - y, ctr[0] - x) + rng.normal(0, 0.6)
        spd = float(rng.uniform(0.6, 1.8))
        v = (spd * math.cos(ang), spd * math.sin(ang), 0.0)
        wv = (-v[1] / r, v[0] / r, 0.0)  # rolling without slipping
        _launch(b, frame_start, v, wv, fps)
        b["p4d_kind"], b["p4d_motion"], b["p4d_class"] = "rigid", "roll", "ball"
        movers.add(b.name)
        summary["rolls"].append(dict(object=b.name, radius_m=r, speed_mps=spd))
    npush = int(rng.integers(push_chairs[0], push_chairs[1] + 1)) if chairs else 0
    for c in list(chairs)[:npush]:  # keyframed push (chairs are big/complex: no simulation)
        mn, mx = world_bbox(c)
        ctr = (mn + mx) / 2
        room_c = (lo + hi) / 2
        away = ctr[:2] - room_c[:2]
        away = away / max(np.linalg.norm(away), 1e-6)
        d = float(rng.uniform(0.2, 0.6))
        yaw = float(rng.normal(0, math.radians(12)))
        t0 = int(rng.integers(frame_start, frame_start + max(1, (frame_end - frame_start) // 3)))
        t1 = min(frame_end, t0 + int(rng.integers(12, 30)))
        p0, r0 = c.location.copy(), c.rotation_euler.copy()
        for f in range(frame_start, frame_end + 1):
            s = np.clip((f - t0) / max(t1 - t0, 1), 0, 1)
            s = s * s * (3 - 2 * s)
            c.location = (p0.x + away[0] * d * s, p0.y + away[1] * d * s, p0.z)
            c.rotation_euler = (r0.x, r0.y, r0.z + yaw * s)
            c.keyframe_insert("location", frame=f)
            c.keyframe_insert("rotation_euler", frame=f)
        c["p4d_kind"], c["p4d_motion"] = "rigid", "push"
        if c.rigid_body is not None:
            c.rigid_body.type = "PASSIVE"
            c.rigid_body.kinematic = True  # moving collider for the simulation
        else:
            _add_rb(c, "PASSIVE", "CONVEX_HULL")
            c.rigid_body.kinematic = True
        movers.add(c.name)
        summary["chairs"].append(dict(object=c.name, dist_m=d, yaw_deg=math.degrees(yaw), frames=[t0, t1]))
    summary["movers"] = sorted(movers)
    return summary


def bake(frame_start, frame_end):
    import bpy

    s = bpy.context.scene
    rw = s.rigidbody_world
    rw.point_cache.frame_start, rw.point_cache.frame_end = frame_start, frame_end
    s.frame_set(frame_start)
    with bpy.context.temp_override(scene=s, point_cache=rw.point_cache):
        bpy.ops.ptcache.free_bake_all()
        bpy.ops.ptcache.bake_all(bake=True)
    s.frame_set(frame_start)


def motion_stats(names, frame_start, frame_end):
    """Per-object world translation / rotation over the clip (after bake)."""
    import bpy

    s = bpy.context.scene
    tr = {n: [] for n in names}
    for f in range(frame_start, frame_end + 1):
        s.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        for n in names:
            o = bpy.data.objects[n].evaluated_get(dg)
            tr[n].append(np.array(o.matrix_world))
    out = {}
    for n, Ms in tr.items():
        Ms = np.array(Ms)
        d = np.linalg.norm(Ms[:, :3, 3] - Ms[0, :3, 3], axis=1)
        R0 = Ms[0, :3, :3] / np.linalg.norm(Ms[0, :3, :3], axis=0)
        R1 = Ms[-1, :3, :3] / np.linalg.norm(Ms[-1, :3, :3], axis=0)
        ang = math.degrees(math.acos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
        out[n] = dict(max_disp_m=float(d.max()), end_disp_m=float(d[-1]), rot_deg=ang)
    s.frame_set(frame_start)
    return out
