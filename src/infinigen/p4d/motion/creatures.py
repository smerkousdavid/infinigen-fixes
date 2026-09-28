# Copyright (C) 2026. This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.
"""Creature locomotion for Infinigen 1.x nature scenes.

Upstream behaviour (3f58bb8): compose_nature spawns ground creatures with animation_mode="idle", but
placement.populate_all() re-instantiates every factory as ``factory_class(seed)`` - the animation mode and
bvh are dropped, so populate builds *unrigged, static* creatures, and the placeholders never move.

This module (opt-in, via infinigen.p4d.nature_driver):
  * `patch_populate(mode_by_factory)`: wraps placement.populate_all so creature factories are re-created
    with the configured animation_mode (default: run for Herbivore/Carnivore, idle for birds/fish/beetles,
    flying birds keep their flight path + wing idle) and a terrain BVH (needed by the idle IK floor snap).
  * `walk_placeholders(...)`: gives ground-creature placeholders a smooth walk over the terrain (speed,
    heading OU noise, slope/radius limits, z snapped to the terrain by raycast, yaw = heading). The creature
    rig built in populate is parented to its placeholder, so gait + path combine.
  * `vertex_motion_report(...)`: gate - evaluated vertices of every creature must move across frames.
"""

from __future__ import annotations

import logging
import math
import re

import numpy as np

logger = logging.getLogger(__name__)

# forward axis of Infinigen v1 creature rigs (gait strides along the armature's +x)
CREATURE_FORWARD_AXIS = "X"

DEFAULT_MODES = {
    "HerbivoreFactory": "run",
    "CarnivoreFactory": "run",
    "BirdFactory": "idle",
    "FlyingBirdFactory": "idle",
    "FishFactory": "idle",
    "BeetleFactory": "walk_cycle",
}
GROUND_WALKERS = ("HerbivoreFactory", "CarnivoreFactory")
SPEED_MPS = {"HerbivoreFactory": (0.8, 1.6), "CarnivoreFactory": (1.0, 2.2)}


def terrain_bvh():
    """BVH of the coarse/fine terrain meshes in the current scene (world space)."""
    import bpy
    from mathutils.bvhtree import BVHTree

    dg = bpy.context.evaluated_depsgraph_get()
    cand = [o for o in bpy.data.objects if o.type == "MESH" and re.search(r"terrain", o.name, re.I)
            and "atmosphere" not in o.name.lower() and "liquid" not in o.name.lower()]
    opaque = [o for o in cand if "opaque" in o.name.lower()] or cand
    if not opaque:
        return None
    verts, faces, off = [], [], 0
    for o in opaque:
        eo = o.evaluated_get(dg)
        me = eo.to_mesh()
        M = np.array(eo.matrix_world)
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        me.calc_loop_triangles()
        tri = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
        verts.append(co)
        faces.append(tri.reshape(-1, 3) + off)
        off += len(co)
        eo.to_mesh_clear()
    V = np.concatenate(verts)
    F = np.concatenate(faces)
    logger.info(f"terrain_bvh from {[o.name for o in opaque]}: {len(V)} verts")
    return BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)


def patch_populate(mode_by_factory=None):
    """Wrap placement.populate_all so creature factories keep an animation mode + bvh at populate time."""
    import inspect

    from infinigen.core.placement import placement
    from infinigen.assets.objects.creatures.util.animation import run_cycle

    if not getattr(run_cycle.animate_run, "_p4d_patched", False):
        original_run = run_cycle.animate_run

        def animate_run(root, arma, targets, **kwargs):
            if root.parent is not None and root.parent.get("p4d_gait"):
                return bake_contact_gait(root, arma, targets, terrain_bvh())
            return original_run(root, arma, targets, **kwargs)

        animate_run._p4d_patched = True
        run_cycle.animate_run = animate_run

    modes = dict(DEFAULT_MODES, **(mode_by_factory or {}))
    orig = placement.populate_all
    if getattr(orig, "_p4d_patched", False):
        return
    cache = {}

    def populate_all(factory_class, cameras, *args, **kwargs):
        name = factory_class.__name__
        if name in modes and "animation_mode" not in kwargs:
            params = inspect.signature(factory_class.__init__).parameters
            if "animation_mode" in params:
                kwargs["animation_mode"] = modes[name]
            if "bvh" in params and "bvh" not in kwargs:
                if "bvh" not in cache:
                    cache["bvh"] = terrain_bvh()
                kwargs["bvh"] = cache["bvh"]
            logger.info(f"p4d: populating {name} with animation_mode={kwargs.get('animation_mode')}")
        return orig(factory_class, cameras, *args, **kwargs)

    populate_all._p4d_patched = True
    # generate_nature calls `placement.populate_all(...)` via the module attribute, so this is enough
    # (do not import generate_nature here: nature_driver runs it with runpy, a 2nd import re-registers gin)
    placement.populate_all = populate_all


def placeholder_factory(obj):
    """'placeholders:HerbivoreFactory(123)' collection -> 'HerbivoreFactory'."""
    for col in obj.users_collection:
        m = re.fullmatch(r"placeholders:(\w+)\(\d*\)", col.name)
        if m:
            return m.group(1)
    return None


def walk_path(rng, start, T, fps, raycast_down, speed_range=(0.8, 1.6), max_radius=12.0, max_slope_deg=32.0,
              heading0=None):
    """Smooth ground walk. raycast_down(x, y) -> (z, normal) or None. Returns pos [T,3], yaw [T], info."""
    dt = 1.0 / fps
    speed = float(rng.uniform(*speed_range))
    yaw = float(heading0 if heading0 is not None else rng.uniform(-math.pi, math.pi))
    yaw_rate = 0.0
    p = np.array(start, np.float64)
    pos, yaws, turned = [], [], 0
    for t in range(T):
        hit = raycast_down(p[0], p[1])
        if hit is not None:
            p[2] = hit[0]
        pos.append(p.copy())
        yaws.append(yaw)
        # OU yaw rate (gentle curving walk), steer back towards start when too far, avoid steep slopes
        yaw_rate = 0.9 * yaw_rate + rng.normal(0, 0.12)
        nxt = p[:2] + speed * dt * np.array([math.cos(yaw), math.sin(yaw)])
        h2 = raycast_down(nxt[0], nxt[1])
        steep = h2 is None or math.degrees(math.acos(max(-1.0, min(1.0, h2[1][2])))) > max_slope_deg
        far = np.linalg.norm(nxt - np.asarray(start)[:2]) > max_radius
        if steep or far:
            back = math.atan2(start[1] - p[1], start[0] - p[0])
            yaw_rate += 0.5 * math.remainder(back - yaw, 2 * math.pi) * dt * 4
            turned += 1
        yaw += yaw_rate * dt
        if not steep:
            p[:2] = nxt
    return np.array(pos), np.unwrap(np.array(yaws)), dict(speed_mps=speed, steer_events=turned)


def walk_placeholders(pois, T, fps, seed=0, frame_start=1, bvh=None):
    """Animate ground-creature placeholders (Herbivore/Carnivore) along terrain walks. Returns info list."""
    from mathutils import Vector

    rng = np.random.default_rng(seed)
    bvh = bvh or terrain_bvh()
    if bvh is None:
        logger.warning("walk_placeholders: no terrain bvh; creatures stay in place")
        return []

    def raycast_down(x, y):
        loc, nrm, _, _ = bvh.ray_cast(Vector((x, y, 1e4)), Vector((0, 0, -1)))
        if loc is None:
            return None
        return loc.z, (nrm.x, nrm.y, nrm.z)

    infos = []
    for o in pois:
        fac = placeholder_factory(o)
        if fac not in GROUND_WALKERS:
            continue
        if o.animation_data is not None and o.animation_data.action is not None:
            continue
        start = np.array(o.matrix_world.translation)
        gait = "walk" if len(infos) % 2 == 0 else "run"
        speed_range = (.35, .7) if gait == "walk" else SPEED_MPS[fac]
        pos, yaw, info = walk_path(rng, start, T, fps, raycast_down, speed_range=speed_range,
                                    max_slope_deg=18)
        base_rot = o.rotation_euler.copy()
        z_off = float(start[2] - (raycast_down(start[0], start[1]) or (start[2],))[0])
        for t in range(T):
            o.location = Vector((pos[t, 0], pos[t, 1], pos[t, 2] + z_off))
            o.rotation_euler = (base_rot.x, base_rot.y, float(yaw[t]))
            o.keyframe_insert("location", frame=frame_start + t)
            o.keyframe_insert("rotation_euler", frame=frame_start + t)
        o["p4d_gait"] = gait
        o["p4d_speed_mps"] = info["speed_mps"]
        info.update(object=o.name, factory=fac, gait=gait,
                    path_len_m=float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()))
        infos.append(info)
        logger.info(f"p4d walk {o.name}: {info}")
    return infos


def bake_contact_gait(root, arma, targets, bvh):
    """Distance-phased IK gait with world-locked stance feet and terrain contacts.

    Root motion is already authored on the placeholder. Targets are baked in
    world space so stance does not slide when the placeholder translates/turns.
    """
    import bpy
    import json
    from mathutils import Vector

    scene = bpy.context.scene
    fs, fe = scene.frame_start, scene.frame_end
    frames = list(range(fs, fe + 1))
    scene.frame_set(fs)
    bpy.context.view_layer.update()
    feet = [o for o in targets if "foot" in o.name.lower()]
    if len(feet) != 4 or bvh is None:
        raise ValueError(f"contact gait requires four feet and terrain: {root.name}, feet={len(feet)}")
    inverse = np.linalg.inv(np.array(root.matrix_world))
    rest = np.array([np.array(o.matrix_world.translation) @ inverse[:3, :3].T + inverse[:3, 3] for o in feet])
    mats = []
    for frame in frames:
        scene.frame_set(frame)
        mats.append(np.array(root.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world))
    mats = np.asarray(mats)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(mats[:, :2, 3], axis=0), axis=1))]
    gait = root.parent["p4d_gait"]
    stride = float(np.clip(np.ptp(rest[:, 0]) * (.45 if gait == "walk" else .7), .25, 1.2))
    duty = .72 if gait == "walk" else .42
    phase = distance / stride
    lift = min(.22, stride * (.18 if gait == "walk" else .3))
    front = rest[:, 0] > np.median(rest[:, 0])
    left = rest[:, 1] > np.median(rest[:, 1])
    reports = []
    for i, foot in enumerate(feet):
        world_pose = foot.matrix_world.copy()
        offset = (0. if front[i] == left[i] else .5) if gait == "run" else (
            0. if front[i] and left[i] else .5 if front[i] else .75 if left[i] else .25)
        phases = phase + offset
        cycles = np.floor(phases).astype(int)
        # Extrapolate beyond clip ends so initial and final partial cycles remain smooth.
        def contact(cycle):
            wanted = (cycle - offset) * stride
            idx = int(np.clip(np.searchsorted(distance, wanted), 0, len(frames) - 1))
            local = rest[i].copy()
            local[0] += stride * duty / 2
            p = mats[idx, :3, :3] @ local + mats[idx, :3, 3]
            tangent = mats[idx, :3, 0] / max(np.linalg.norm(mats[idx, :3, 0]), 1e-8)
            p += tangent * (wanted - distance[idx])
            hit, _, _, _ = bvh.ray_cast(Vector((p[0], p[1], p[2] + 5)), Vector((0, 0, -1)))
            if hit is None:
                raise ValueError(f"foot contact misses terrain: {root.name}")
            p[2] = hit.z + .01
            return p

        contacts = {c: contact(c) for c in range(int(cycles.min()), int(cycles.max()) + 2)}
        values, stance = [], []
        for cycle, ph in zip(cycles, phases % 1):
            planted = ph < duty
            if planted:
                p = contacts[int(cycle)].copy()
            else:
                u = (ph - duty) / (1 - duty)
                ease = u * u * (3 - 2 * u)
                p = (1 - ease) * contacts[int(cycle)] + ease * contacts[int(cycle) + 1]
                p[2] += lift * np.sin(np.pi * u)
            values.append(p)
            stance.append(bool(planted))
        foot.parent = None
        foot.matrix_world = world_pose
        foot.animation_data_clear()
        for constraint in list(foot.constraints):
            foot.constraints.remove(constraint)
        for frame, p in zip(frames, values):
            foot.location = p
            foot.keyframe_insert("location", frame=frame)
        values = np.asarray(values)
        stable = np.asarray(stance[1:]) & np.asarray(stance[:-1]) & (np.diff(cycles) == 0)
        slip = np.linalg.norm(np.diff(values, axis=0), axis=1)[stable]
        bones = [bone.name for bone in arma.pose.bones for con in bone.constraints
                 if con.type == "IK" and con.target == foot]
        if len(bones) != 1:
            raise ValueError(f"ambiguous IK endpoint for {foot.name}: {bones}")
        reports.append(dict(target=foot.name, bone=bones[0], stance=stance, cycles=cycles.tolist(),
                            target_world=values.tolist(),
                            stance_slip_max_m=float(slip.max(initial=0))))
    report = dict(gait=gait, armature=arma.name, stride_m=stride, duty_factor=duty,
                  root_distance_m=float(distance[-1]), feet=reports)
    report["evaluated_contacts"] = evaluate_contacts(report, frames)
    logger.info("p4d evaluated %s gait: %s", gait, report["evaluated_contacts"])
    root["p4d_gait_report"] = json.dumps(report)
    root["p4d_kind"] = "creature"
    scene.frame_set(fs)
    return []


def evaluate_contacts(gait, frames):
    """Check evaluated IK endpoints, independently of the authored target arrays."""
    import bpy
    scene = bpy.context.scene
    arma = bpy.data.objects[gait["armature"]]
    samples = {foot["bone"]: [] for foot in gait["feet"]}
    for frame in frames:
        scene.frame_set(frame)
        evaluated = arma.evaluated_get(bpy.context.evaluated_depsgraph_get())
        for foot in gait["feet"]:
            position = evaluated.matrix_world @ evaluated.pose.bones[foot["bone"]].tail
            samples[foot["bone"]].append(list(position))
    errors, slips = [], []
    for foot in gait["feet"]:
        actual = np.asarray(samples[foot["bone"]])
        stance = np.asarray(foot["stance"])
        errors.extend(np.linalg.norm(actual - foot["target_world"], axis=1)[stance].tolist())
        stable = stance[1:] & stance[:-1] & (np.diff(foot["cycles"]) == 0)
        slips.extend(np.linalg.norm(np.diff(actual, axis=0), axis=1)[stable].tolist())
    return dict(endpoint_target_error_p95_m=float(np.percentile(errors, 95)) if errors else None,
                stance_step_slip_p95_m=float(np.percentile(slips, 95)) if slips else None,
                stance_samples=len(errors), stable_steps=len(slips), frames_checked=len(frames))


def vertex_motion_report(objects, frames, depsgraph=None):
    """Max / median vertex displacement (world) of each object's evaluated mesh between the given frames."""
    import bpy

    scene = bpy.context.scene
    rep = {}
    snaps = {o.name: [] for o in objects}
    topologies = {o.name: [] for o in objects}
    for f in frames:
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        for o in objects:
            eo = o.evaluated_get(dg)
            try:
                me = eo.to_mesh()
            except RuntimeError:
                continue
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            me.calc_loop_triangles()
            tri = np.empty(len(me.loop_triangles) * 3, np.int32)
            me.loop_triangles.foreach_get("vertices", tri)
            topologies[o.name].append(tri)
            M = np.array(eo.matrix_world)
            snaps[o.name].append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
            eo.to_mesh_clear()
    for name, s in snaps.items():
        if len(s) < 2 or any(len(x) != len(s[0]) for x in s) or any(
                not np.array_equal(tr, topologies[name][0]) for tr in topologies[name]):
            rep[name] = {"topology_constant": False}
            continue
        # max over all checked frames (profiles like open-then-close return to the start pose)
        d = np.max([np.linalg.norm(x - s[0], axis=1) for x in s[1:]], axis=0)
        # Remove best-fitting rigid rotation AND translation; turning is not a gait.
        dl = np.max([rigid_residual(s[0], x) for x in s[1:]], axis=0)
        rep[name] = {"topology_constant": True, "n_verts": int(len(d)), "max_disp_m": float(d.max()),
                     "median_disp_m": float(np.median(d)), "max_local_deform_m": float(dl.max())}
    return rep


def rigid_residual(reference, points):
    """Per-vertex nonrigid displacement after a proper Kabsch alignment."""
    a = np.asarray(reference, np.float64) - np.mean(reference, axis=0)
    b = np.asarray(points, np.float64) - np.mean(points, axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return np.linalg.norm(a @ (u @ correction @ vt) - b, axis=1)
