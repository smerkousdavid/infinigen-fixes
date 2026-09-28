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
import hashlib

import numpy as np

logger = logging.getLogger(__name__)


def world_bbox(o):
    import bpy

    dg = bpy.context.evaluated_depsgraph_get()
    eo = o.evaluated_get(dg)
    M = np.array(eo.matrix_world)
    mesh = eo.to_mesh()
    mesh.calc_loop_triangles()
    if len(mesh.loop_triangles):
        # Sim assets carry loose vertices/edges encoding joint axes and limits.
        # They are not rendered surfaces and must not inflate collision bounds.
        triangles = np.empty(len(mesh.loop_triangles) * 3, np.int32)
        mesh.loop_triangles.foreach_get("vertices", triangles)
        vertices = np.empty(len(mesh.vertices) * 3, np.float64)
        mesh.vertices.foreach_get("co", vertices)
        bb = vertices.reshape(-1, 3)[np.unique(triangles)]
    else:
        bb = np.array([list(v) for v in eo.bound_box])
    eo.to_mesh_clear()
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
    # Keep a nonzero derivative at release. Blender's default constant
    # extrapolation after the last transform key otherwise cancels the impulse.
    o.location = p0 + type(p0)(v) * (2 * dt)
    o.rotation_euler = (r0.x + w[0] * 2 * dt, r0.y + w[1] * 2 * dt, r0.z + w[2] * 2 * dt)
    o.keyframe_insert("location", frame=frame_start + 2)
    o.keyframe_insert("rotation_euler", frame=frame_start + 2)
    for curve in o.animation_data.action.fcurves:
        curve.extrapolation = "LINEAR"
        for key in curve.keyframe_points:
            key.interpolation = "CONSTANT" if "kinematic" in curve.data_path else "LINEAR"
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
        color_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        c = np.random.default_rng(color_seed).uniform(0.1, 0.9, 3)
        chk.inputs["Color1"].default_value = (*c, 1)
        chk.inputs["Color2"].default_value = (*(1 - c), 1)
        mat.node_tree.links.new(chk.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.5
    o.data.materials.append(mat)
    return o


def chair_sweep_clear(chair, obstacles, away, distance, yaw, frame_count):
    """Test actual chair surfaces: a tucked chair overlaps the table's AABB."""
    import bpy
    from mathutils.bvhtree import BVHTree
    from infinigen.p4d import cameras as C
    from infinigen.p4d.gt import eval_mesh

    _, _, scene_bvh = C.blender_bvh_callbacks(obstacles)
    co, tri = eval_mesh(chair, bpy.context.evaluated_depsgraph_get())
    origin = np.asarray(chair.matrix_world.translation)
    for s in np.linspace(0, 1, max(frame_count, 2)):
        angle = yaw * s
        c, sn = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.]])
        points = (co - origin) @ rotation.T + origin + np.r_[away * distance * s, 0]
        chair_bvh = BVHTree.FromPolygons(points.tolist(), tri.tolist(), all_triangles=True)
        if chair_bvh.overlap(scene_bvh):
            return False
    return True


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
    if len(small) < 2:
        raise ValueError("physics scene needs at least two supported small objects for drop and slide")
    nd = min(nd, len(small) - 1)
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
        headroom = float(hi[2] - mx[2] - .1)
        if headroom < .3:
            o.rigid_body.type = "PASSIVE"
            continue
        h = float(rng.uniform(.3, min(1.2, headroom)))
        o.location.z += h
        v = (float(rng.normal(0, 0.6)), float(rng.normal(0, 0.6)), float(rng.uniform(-0.5, 1.0)))
        w = tuple(float(x) for x in rng.normal(0, 3.0, 3))
        _launch(o, frame_start, v, w, fps)
        o["p4d_kind"], o["p4d_motion"] = "rigid", "drop"
        movers.add(o.name)
        summary["drops"].append(dict(object=o.name, lift_m=h, v=v, w=w, mass=mass,
                                     actuation="initial release and impulse", friction=o.rigid_body.friction,
                                     restitution=o.rigid_body.restitution))
    for o in slides:  # push along its support surface (it will slide / tip / fall off edges)
        mn, mx = world_bbox(o)
        if np.max(mx - mn) > 0.8:
            continue
        mass = float(np.clip(300 * np.prod(np.maximum(mx - mn, .02)), .05, 5.))
        _add_rb(o, "ACTIVE", "CONVEX_HULL", mass=mass, friction=float(rng.uniform(0.2, 0.45)), restitution=0.05)
        ang = rng.uniform(0, 2 * math.pi)
        spd = float(rng.uniform(0.6, 1.6))
        v = (spd * math.cos(ang), spd * math.sin(ang), 0.0)
        o.location.z += 0.003
        w = (0.0, 0.0, float(rng.normal(0, 1.0)))
        _launch(o, frame_start, v, w, fps)
        o["p4d_kind"], o["p4d_motion"] = "rigid", "slide"
        movers.add(o.name)
        summary["slides"].append(dict(object=o.name, speed_mps=spd, heading_deg=math.degrees(ang),
                                      actuation="initial horizontal impulse", mass=mass, v=v, w=w,
                                      friction=o.rigid_body.friction))
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
        else:
            bpy.data.objects.remove(b, do_unlink=True)
            raise ValueError("could not find an unoccupied support for a rolling ball")
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
        summary["rolls"].append(dict(object=b.name, radius_m=r, speed_mps=spd,
                                    actuation="initial rolling impulse", mass=b.rigid_body.mass, v=v, w=wv))
    npush = int(rng.integers(push_chairs[0], push_chairs[1] + 1)) if chairs else 0
    for c in chairs:  # try all available chairs until the requested number fits
        if len(summary["chairs"]) >= npush:
            _add_rb(c, "PASSIVE", "MESH")
            continue
        mn, mx = world_bbox(c)
        ctr = (mn + mx) / 2
        room_c = (lo + hi) / 2
        tables = [sum(world_bbox(o)) / 2 for o in scene_objects if "dining_table" in o.name]
        support_centre = min(tables, key=lambda p: np.linalg.norm(p[:2] - ctr[:2])) if tables else room_c
        away = ctr[:2] - support_centre[:2]
        away = away / max(np.linalg.norm(away), 1e-6)
        d = float(rng.uniform(0.2, 0.6))
        yaw = 0.0  # straight pull from the table; avoid sweeping chair legs sideways
        t0 = int(rng.integers(frame_start, frame_start + max(1, (frame_end - frame_start) // 3)))
        t1 = min(frame_end, t0 + int(rng.integers(12, 30)))
        p0, r0 = c.location.copy(), c.rotation_euler.copy()
        # A kinematic chair cannot resolve interpenetrations: reject its swept AABB first.
        swept_lo = mn + np.r_[np.minimum(0, away * d), 0]
        swept_hi = mx + np.r_[np.maximum(0, away * d), 0]
        yaw_pad = .5 * np.linalg.norm((mx - mn)[:2]) * abs(yaw)
        swept_lo[:2] -= yaw_pad
        swept_hi[:2] += yaw_pad
        obstacles = [o for o in scene_objects if o.type == "MESH" and o != c and
                     not o.name.startswith(("room_floor", "room_skirting", "rug.")) and o.name not in chosen]
        if np.any(swept_lo[:2] < lo[:2] + .02) or np.any(swept_hi[:2] > hi[:2] - .02) or not chair_sweep_clear(
                c, obstacles, away, d, yaw, frame_end - frame_start + 1):
            logger.info(f"rejecting obstructed chair push: {c.name}")
            _add_rb(c, "PASSIVE", "MESH")
            continue
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
        summary["chairs"].append(dict(object=c.name, dist_m=d, yaw_deg=math.degrees(yaw), frames=[t0, t1],
                                      actuation="externally pushed kinematic chair", swept_clearance_checked=True))
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


def bake_explicit(frame_start, frame_end, summary, fps=24, substeps=20):
    """CPU Bullet with explicit velocities, then bake its poses into Blender.

    Blender resets velocity at the kinematic/dynamic handoff. PyBullet exposes
    resetBaseVelocity, so drops, slides and rolling impulses are reproducible.
    Both render geometry and tracks subsequently evaluate the same baked poses.
    """
    import bpy
    import pybullet as bullet
    from mathutils import Matrix
    from infinigen.p4d.gt import eval_mesh

    scene = bpy.context.scene
    frames = list(range(frame_start, frame_end + 1))
    scene.frame_set(frame_start)
    scene.rigidbody_world.enabled = False
    moving = set(summary["movers"])
    chairs = {r["object"] for r in summary["chairs"]}
    dynamic = moving - chairs
    records = {r["object"]: r for k in ("drops", "slides", "rolls") for r in summary[k]}
    objects = [o for o in scene.objects if o.type == "MESH" and o.rigid_body is not None]
    client = bullet.connect(bullet.DIRECT)
    bodies, tracks = {}, {n: [] for n in dynamic}
    try:
        bullet.setGravity(0, 0, -9.81, physicsClientId=client)
        bullet.setTimeStep(1 / (fps * substeps), physicsClientId=client)
        bullet.setPhysicsEngineParameter(numSolverIterations=80, deterministicOverlappingPairs=1,
                                         physicsClientId=client)
        for obj in objects:
            co, tri = eval_mesh(obj, bpy.context.evaluated_depsgraph_get())
            if not len(tri):
                continue
            centre = (co[np.unique(tri)].min(0) + co[np.unique(tri)].max(0)) / 2
            mass = obj.rigid_body.mass if obj.name in dynamic else 0
            shape_args = dict(shapeType=bullet.GEOM_MESH, vertices=(co - centre).tolist())
            if obj.name in records and "radius_m" in records[obj.name]:
                shape_args = dict(shapeType=bullet.GEOM_SPHERE, radius=records[obj.name]["radius_m"])
            elif mass == 0:
                shape_args.update(indices=tri.ravel().tolist(), flags=bullet.GEOM_FORCE_CONCAVE_TRIMESH)
            collision = bullet.createCollisionShape(**shape_args, physicsClientId=client)
            body = bullet.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision,
                                          basePosition=centre.tolist(), physicsClientId=client)
            rb = obj.rigid_body
            bullet.changeDynamics(body, -1, lateralFriction=rb.friction, restitution=rb.restitution,
                                  linearDamping=rb.linear_damping, angularDamping=rb.angular_damping,
                                  collisionMargin=.002, physicsClientId=client)
            bodies[obj.name] = (body, centre, np.asarray(obj.matrix_world).copy())
            if mass:
                record = records[obj.name]
                if "v" in record:
                    velocity, spin = record["v"], record["w"]
                else:
                    if "heading_deg" in record:
                        angle = np.radians(record["heading_deg"])
                        velocity = np.r_[record["speed_mps"] * np.array([np.cos(angle), np.sin(angle)]), 0]
                        spin = [0, 0, 0]
                    else:
                        velocity, spin = record["v"], record["w"]
                bullet.resetBaseVelocity(body, linearVelocity=velocity, angularVelocity=spin,
                                         physicsClientId=client)
        for frame in frames:
            scene.frame_set(frame)
            for name in chairs:
                body, centre, original = bodies[name]
                pose = np.asarray(bpy.data.objects[name].matrix_world) @ np.linalg.inv(original)
                rot = Matrix(pose[:3, :3].tolist()).to_quaternion()
                pos = pose[:3, :3] @ centre + pose[:3, 3]
                bullet.resetBasePositionAndOrientation(body, pos, (rot.x, rot.y, rot.z, rot.w), physicsClientId=client)
            if frame != frame_start:
                for _ in range(substeps):
                    bullet.stepSimulation(physicsClientId=client)
            for name in dynamic:
                body, centre, original = bodies[name]
                pos, quat = bullet.getBasePositionAndOrientation(body, physicsClientId=client)
                rot = np.asarray(bullet.getMatrixFromQuaternion(quat)).reshape(3, 3)
                delta = np.eye(4)
                delta[:3, :3], delta[:3, 3] = rot, np.asarray(pos) - rot @ centre
                tracks[name].append(delta @ original)
        for name, matrices in tracks.items():
            obj = bpy.data.objects[name]
            obj.animation_data_clear()
            obj.rotation_mode = "QUATERNION"
            for frame, matrix in zip(frames, matrices):
                obj.matrix_world = Matrix(matrix.tolist())
                obj.keyframe_insert("location", frame=frame)
                obj.keyframe_insert("rotation_quaternion", frame=frame)
            for curve in obj.animation_data.action.fcurves:
                for key in curve.keyframe_points:
                    key.interpolation = "LINEAR"
        summary["simulation"] = dict(engine="PyBullet", api_version=bullet.getAPIVersion(),
                                      fps=fps, substeps=substeps, gravity_mps2=[0, 0, -9.81],
                                      initial_velocity="explicit", poses="baked to Blender world transforms")
    finally:
        bullet.disconnect(client)
        scene.frame_set(frame_start)


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
