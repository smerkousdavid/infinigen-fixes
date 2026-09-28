# Copyright (C) 2026. This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.
"""Drop-in driver for infinigen_examples.generate_nature with p4d motion + multi-view cameras.

    python -m infinigen.p4d.nature_driver <same args as generate_nature> [--p4d_views 4] [--p4d_seed S]

Patches (all opt-in, upstream files untouched):
  * creature populate keeps animation_mode + terrain bvh (motion/creatures.patch_populate)
  * cam_traj.animate_trajectories -> `animate_multiview`: first walks ground-creature placeholders over the
    terrain, then gives each of the V camera rigs its own path (infinigen.p4d.cameras), validated against the
    coarse scene BVH. Per-view metadata is stored on the rig (custom prop "p4d_view", JSON).
  * registers the gin configurable `p4d_render_image` (infinigen.p4d.gt) for
    `-p render.render_image_func=@p4d_render_image`.
Use with `-p execute_tasks.frame_range=[1,T] camera.spawn_camera_rigs.n_camera_rigs=V` and
`camera.spawn_camera_rigs.camera_rig_config=[{'loc': (0, 0, 0), 'rot_euler': (0, 0, 0)}]`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import runpy
import sys

from infinigen.p4d.runtime import configure_cpu_budget
configure_cpu_budget()

import numpy as np

logger = logging.getLogger(__name__)

P4D = {"views": 4, "seed": 0, "creature_view_frac": 0.7, "jitter_prob": 0.25, "overlap": "none", "fps": 24}


def _pop_arg(name, default, cast):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = cast(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        return v
    return default


def _moving_targets(pois, T, frame_start):
    """World trajectories [T,3] of animated points of interest (creature placeholders, flying birds)."""
    import bpy

    scene = bpy.context.scene
    tr = {o.name: [] for o in pois}
    for t in range(T):
        scene.frame_set(frame_start + t)
        for o in pois:
            tr[o.name].append(np.array(o.matrix_world.translation))
    out = {}
    for o in pois:
        a = np.array(tr[o.name])
        if np.linalg.norm(a[-1] - a[0]) > 0.3 or np.ptp(a, axis=0).max() > 0.3:
            out[o.name] = a
    scene.frame_set(frame_start)
    return out


def animate_multiview(cam_rigs, base_views, scene_preprocessed, obj_groups=None, pois=None, **kw):
    import bpy
    from mathutils import Vector

    from infinigen.p4d import cameras as C
    from infinigen.p4d.motion import creatures

    scene = bpy.context.scene
    fs, fe = scene.frame_start, scene.frame_end
    T, fps = fe - fs + 1, float(scene.render.fps)
    rng = np.random.default_rng(P4D["seed"])
    pois = list(pois or [])
    bvh = scene_preprocessed["scene_bvh"]
    tbvh = creatures.terrain_bvh()
    walks = creatures.walk_placeholders(pois, T, fps, seed=P4D["seed"], frame_start=fs, bvh=tbvh)
    moving = _moving_targets(pois, T, fs)
    logger.info(f"p4d: {len(walks)} walking creatures, {len(moving)} moving pois")

    def clearance_fn(points):
        out = []
        for p in points:
            hit = bvh.find_nearest(Vector(p))
            out.append(np.inf if hit[0] is None else hit[3])
        return np.array(out)

    def ray_fn(origins, dirs):
        out = []
        for o, d in zip(origins, dirs):
            hit = bvh.ray_cast(Vector(o), Vector(d))
            out.append(np.inf if hit[0] is None else hit[3])
        return np.array(out)

    def ground_z(x, y):
        loc, *_ = (tbvh or bvh).ray_cast(Vector((x, y, 1e4)), Vector((0, 0, -1)))
        return None if loc is None else loc.z

    def above_ground(points):
        heights = [ground_z(*p[:2]) for p in points]
        return np.array([z is not None and p[2] - z >= .3 for p, z in zip(points, heights)])

    anchors = []
    for v in base_views:
        _, prop, _ = v
        anchors.append((np.array(prop.loc), np.array(prop.rot)))
    if not anchors:
        anchors = [(np.array(r.location), np.array(r.rotation_euler)) for r in cam_rigs]

    if P4D["overlap"] != "none":
        from infinigen.p4d.rigs import sample_rig, TIERS

        tiers = TIERS if P4D["overlap"] == "all" else (P4D["overlap"],)
        if len(cam_rigs) != 4 * len(tiers):
            raise ValueError(f"expected {4 * len(tiers)} camera rigs, got {len(cam_rigs)}")
        if moving:
            from infinigen.p4d.motion.objects import world_bbox
            scene.frame_set(fs)
            heights = {o.name: float(np.ptp(np.array(world_bbox(o)), axis=0)[2])
                       for o in pois if o.name in moving}
            actor_targets = {name: path + [0, 0, .25 * heights[name]] for name, path in moving.items()}
            target_name = "creature_group_center"
            target = np.mean(list(actor_targets.values()), axis=0)
            minimum_radius = max(7., 1.8 * max(heights.values()))
        else:
            anchor, rot = anchors[0]
            forward = np.array(_euler_to_forward(rot))
            loc, *_ = bvh.ray_cast(Vector(anchor), Vector(forward), 30.)
            centre = np.array(loc) if loc is not None else anchor + 8 * forward
            gz = ground_z(*centre[:2])
            if gz is not None:
                centre[2] = gz + 1
            target, target_name = np.tile(centre, (T, 1)), "vegetation_region"

        def anchor_fn(rng):
            for _ in range(200):
                az = rng.uniform(-np.pi, np.pi)
                radius = rng.uniform(minimum_radius, minimum_radius + 5) if moving else rng.uniform(3, 8)
                xy = target[0, :2] + radius * np.array([np.cos(az), np.sin(az)])
                gz = ground_z(*xy)
                if gz is None:
                    continue
                height = gz + rng.uniform(1.2, 2.)
                if moving:
                    height = max(gz + rng.uniform(1.7, 2.5), target[0, 2] + .5)
                    if height - gz > 3.5:
                        continue
                p = np.r_[xy, height]
                if clearance_fn([p])[0] > .4:
                    return p
            raise ValueError("no clear nature camera anchor")

        metas = []
        for k, tier in enumerate(tiers):
            rng = np.random.default_rng(np.random.SeedSequence([P4D["seed"], TIERS.index(tier), 193]))
            rigs = cam_rigs[k * 4:(k + 1) * 4]
            intrs = []
            lens = float(rng.choice([24, 35] if moving else [24, 35, 50]))
            for rig in rigs:
                cam = rig.children[0]
                cam.data.lens, cam.data.sensor_width, cam.data.sensor_fit = lens, 36., "HORIZONTAL"
                # Native terrain culling additionally requires matching sensor
                # aspect, even though Blender's horizontal-fit projection does not.
                cam.data.sensor_height = 36. * scene.render.resolution_y / scene.render.resolution_x
                cam.data.dof.use_dof = False
                intrs.append(C.Intrinsics.from_camera(cam, scene, bpy.context.evaluated_depsgraph_get()))
            tier_targets, target_names = [target] * 4, [target_name] * 4
            if tier == "low":
                # Look across different parts of the same scene. Opposing views
                # of an open ground plane can still have high surface overlap.
                tier_targets, target_names = [], []
                for i, offset in enumerate(([6, 0, 0], [-6, 0, 0], [0, 6, 0], [0, -6, 0])):
                    if i < len(moving):
                        name = list(moving)[i]
                        region = actor_targets[name]
                    else:
                        centre = target[0] + offset
                        gz = ground_z(*centre[:2])
                        if gz is None:
                            raise ValueError("low-overlap target misses terrain")
                        centre[2] = gz + .7
                        region, name = np.tile(centre, (T, 1)), f"vegetation_region_{i}"
                    tier_targets.append(region)
                    target_names.append(name)
            sampled = sample_rig(rng, T, fps, intrs, anchor_fn, target, tier, clearance_fn, ray_fn,
                                 inside_fn=above_ground, view_targets=tier_targets,
                                 max_tries=400 if tier == "low" else 160)
            for i, (rig, view) in enumerate(zip(rigs, sampled)):
                C.blender_apply_path(rig, view["pos"], view["R"], frame_start=fs)
                meta = dict(view["meta"], view_id=i, seed=int(P4D["seed"] * 100 + k * 4 + i),
                            target_object=target_names[i], lens_mm=lens)
                rig["p4d_view"] = json.dumps(meta)
                metas.append(meta)
        scene["p4d_motion"] = json.dumps({"creature_walks": walks, "moving_pois": list(moving)})
        scene["p4d_views"] = json.dumps(metas)
        return metas

    cam0 = cam_rigs[0].children[0]
    intr = C.Intrinsics.from_blender(cam0.data.lens, cam0.data.sensor_width, scene.render.resolution_x,
                                     scene.render.resolution_y)
    # ground walkers first (the interesting motion); flying creatures only as a fallback target
    ground = {w["object"] for w in walks}
    movers = [(k, v) for k, v in moving.items() if k in ground]
    rng.shuffle(movers)
    fly = [(k, v) for k, v in moving.items() if k not in ground]
    rng.shuffle(fly)
    movers += fly[: max(0, len(cam_rigs) - len(movers))] if not movers else []
    types = C.assign_path_types(rng, len(cam_rigs), has_target=True)
    metas = []
    for i, rig in enumerate(cam_rigs):
        pt = types[i]
        target = None
        tgt_name = None
        use_creature = movers and (pt in ("follow", "orbit") or rng.random() < P4D["creature_view_frac"])
        if use_creature:
            tgt_name, traj = movers[i % len(movers)]
            target = traj + np.array([0, 0, 0.6])
            # anchor: 4-10 m from the creature's start, 1.2-3 m above the ground
            for _ in range(30):
                d, az = rng.uniform(4, 10), rng.uniform(-math.pi, math.pi)
                x, y = target[0, 0] + d * math.cos(az), target[0, 1] + d * math.sin(az)
                gz = ground_z(x, y)
                if gz is not None:
                    anchor = np.array([x, y, gz + rng.uniform(1.2, 3.0)])
                    break
            else:
                anchor = anchors[i % len(anchors)][0]
            scale = float(np.clip(np.linalg.norm(anchor - target[0]), 3, 12))
        else:
            anchor, rot = anchors[i % len(anchors)]
            # look target = where the base view looks (ray hit), else 10 m ahead
            fwd = np.array(_euler_to_forward(rot))
            loc, *_ = bvh.ray_cast(Vector(anchor), Vector(fwd), 60.0)
            tp = np.array(loc) if loc is not None else anchor + 10 * fwd
            target = tp if pt in C.NEEDS_TARGET else (tp if rng.random() < 0.5 else None)
            scale = float(np.clip(np.linalg.norm(tp - anchor), 3, 15))
        view = C.sample_view(rng, T, fps, intr, anchor, scale, target=target, path_type=pt,
                             jitter_prob=P4D["jitter_prob"], clearance_fn=clearance_fn, ray_fn=ray_fn,
                             max_miss_frac=0.6)
        C.blender_apply_path(rig, view["pos"], view["R"], frame_start=fs)
        meta = dict(view["meta"], view_id=i, seed=int(P4D["seed"] * 100 + i), target_object=tgt_name,
                    anchor=anchor.tolist(), scale_m=scale)
        rig["p4d_view"] = json.dumps(meta)
        metas.append(meta)
        logger.info(f"p4d view {i}: {meta['path_type']} jitter={meta['jitter']} attempts={meta['attempts']} "
                    f"target={tgt_name} valid={meta['validation']}")
    scene["p4d_motion"] = json.dumps({"creature_walks": walks, "moving_pois": list(moving)})
    scene["p4d_views"] = json.dumps(metas)
    return metas


def _euler_to_forward(rot):
    from mathutils import Euler, Vector

    return tuple(Euler(tuple(rot), "XYZ").to_matrix() @ Vector((0, 0, -1)))


def main():
    P4D["views"] = _pop_arg("--p4d_views", P4D["views"], int)
    P4D["seed"] = _pop_arg("--p4d_seed", P4D["seed"], int)
    P4D["jitter_prob"] = _pop_arg("--p4d_jitter_prob", P4D["jitter_prob"], float)
    P4D["overlap"] = _pop_arg("--p4d_overlap", P4D["overlap"], str)
    P4D["fps"] = _pop_arg("--p4d_fps", P4D["fps"], int)
    os.environ.setdefault("INFINIGEN_DISABLE_SLURM", "1")

    import bpy
    from mathutils import noise
    # Only this driver loads blendfiles generated by this pipeline.
    bpy.context.preferences.filepaths.use_scripts_auto_execute = True
    bpy.app.driver_namespace["noise"] = noise
    bpy.context.scene.render.fps = P4D["fps"]

    from infinigen.core.placement import camera_trajectories as cam_traj
    from infinigen.p4d import gt  # noqa: F401  (registers @p4d_render_image)
    from infinigen.p4d.motion import creatures
    from infinigen.p4d.visibility import patch_population_visibility

    creatures.patch_populate()
    patch_population_visibility()
    if P4D["overlap"] != "none":
        original_poses = cam_traj.compute_poses

        def initial_pose(cam_rigs, **kwargs):
            # The curriculum sampler authors every final camera. Only one native
            # seed view is needed to choose a terrain region, avoiding twelve
            # expensive searches whose results would immediately be discarded.
            return original_poses(cam_rigs=cam_rigs[:1], **kwargs)

        cam_traj.compute_poses = initial_pose
    cam_traj.animate_trajectories = animate_multiview
    sys.argv[0] = "infinigen_examples.generate_nature"
    runpy.run_module("infinigen_examples.generate_nature", run_name="__main__", alter_sys=True)
    # bpy 4.2 can segfault while destructing baked particle caches at Python
    # shutdown, after the stage has saved successfully. Only bypass teardown
    # after the entire generator returns normally; exceptions still fail.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
