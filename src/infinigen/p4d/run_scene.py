# Copyright (C) 2026. This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.
"""One p4d multi-view scene end to end.

    python -m infinigen.p4d.run_scene --family FAMILY --seed S --out DIR [--frames 48] [--width 640 --height 360]
                                      [--views 4] [--samples 128]

Families
  nature_creatures  v1 nature (desert/plain), walking herbivores/carnivores (run gait on terrain walks), mild wind
  nature_wind       v1 nature (forest/plain), strong wind on trees/bushes/grass, falling leaves
  indoor_physics    v2 furnished room, rigid-body drops/tosses/slides, rolling balls, pushed chairs
  indoor_artic      v2 furnished room + 2-3 Infinigen-Sim articulated assets with animated joints
  indoor_flying     v2 flying_indoor floating random walk (kept at low weight)

Output: DIR/<family>_sSSSSS (raw multi-view scene, see infinigen.p4d.gt) + DIR/run.json (stage timings, commands).
The nature families run the v1 stages as subprocesses through infinigen.p4d.nature_driver; indoor families run
in this process (bpy).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("p4d.run_scene")

FAMILIES = ("nature_creatures", "nature_wind", "indoor_physics", "indoor_artic", "indoor_flying")


def raw_name(a):
    """raw scene dir name == p4d clip key (converter uses the dir name)."""
    return f"{a.family}_s{a.seed:05d}"


# ------------------------------------------------------------------------------------------------ nature (v1)
def nature_gin(family, rng, a):
    if family == "nature_creatures":
        scene = "desert.gin"  # plain/forest coarse+populate take >1 h on 8 vCPUs; desert ~12 min
        wind = float(rng.uniform(0.015, 0.035))
        extra = ["compose_nature.ground_creatures_chance=1.0", "compose_nature.max_ground_creatures=2",  # rig weights ~7 min per creature on 8 vCPUs
                 "compose_nature.ground_creature_registry=[(@CarnivoreFactory,1),(@HerbivoreFactory,1.5)]",
                 "compose_nature.flying_creatures_chance=0.5", "compose_nature.center_distance=15"]
        flutter = 0.0
    else:
        scene = str(rng.choice(["forest.gin", "plain.gin"], p=[0.5, 0.5]))
        wind = float(rng.uniform(0.05, 0.1))
        flutter = float(rng.uniform(0.01, 0.03))
        extra = ["compose_nature.leaf_particles_chance=1.0", "compose_nature.ground_creatures_chance=0.0",
                 "compose_nature.flying_creatures_chance=0.0", "compose_nature.trees_chance=1.0"]
    common = [f"execute_tasks.frame_range=[1,{a.frames}]", f"execute_tasks.generate_resolution=({a.width},{a.height})",
              f"camera.spawn_camera_rigs.n_camera_rigs={a.views}",
              "camera.spawn_camera_rigs.camera_rig_config=[{'loc': (0, 0, 0), 'rot_euler': (0, 0, 0)}]",
              "compose_nature.inview_distance=35", "placement.populate_all.dist_cull=30",
              # SphericalMesher asserts fov < 90 deg for every camera; OcMesher handles V wide cameras (and is what
              # upstream uses for video)
              'fine_terrain.mesher_backend="OcMesher"']
    return scene, wind, flutter, common + extra


def run_nature(a, out):
    rng = np.random.default_rng(a.seed)
    scene, wind, flutter, over = nature_gin(a.family, rng, a)
    cfg = [scene, "dev.gin", "fast_terrain_assets.gin"]
    py = sys.executable
    drv = [py, "-m", "infinigen.p4d.nature_driver", "--p4d_views", str(a.views), "--p4d_seed", str(a.seed)]
    base = ["--seed", str(a.seed), "-g", *cfg, "-p", *over]
    raw = out / raw_name(a)
    render_over = ["render.render_image_func=@p4d_render_image", f"p4d_render_image.family='{a.family}'",
                   f"p4d_render_image.seed={a.seed}", f"p4d_render_image.out_dir='{raw}'",
                   f"p4d_render_image.samples={a.samples}", f"p4d_render_image.wind_strength={wind}",
                   f"p4d_render_image.wind_flutter={flutter}", "p4d_render_image.wind_gust=0.6"]
    stages = [
        ("coarse", ["--task", "coarse", "--output_folder", str(out / "coarse")], []),
        ("populate_fine", ["--task", "populate", "fine_terrain", "--input_folder", str(out / "coarse"),
                           "--output_folder", str(out / "fine")], []),
        ("render_gt", ["--task", "render", "--input_folder", str(out / "fine"), "--output_folder",
                       str(out / "frames")], render_over),
    ]
    info = dict(family=a.family, scene_config=scene, wind_strength=wind, flutter=flutter, gin_overrides=over,
                stages={})
    env = dict(os.environ, INFINIGEN_DISABLE_SLURM="1")
    for name, args, extra_p in stages:
        cmd = drv + base[:] + args
        if extra_p:
            cmd = cmd + extra_p  # appended to the -p list (argparse nargs='+' keeps collecting)
        t0 = time.time()
        with open(out / f"{name}.log", "w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(out))
        info["stages"][name] = dict(wall_s=round(time.time() - t0, 1), exit=r.returncode, cmd=cmd)
        logger.info(f"{name}: exit {r.returncode} in {time.time() - t0:.0f}s")
        (out / "run.json").write_text(json.dumps(info, indent=1))
        if r.returncode != 0:
            raise RuntimeError(f"stage {name} failed (exit {r.returncode}); see {out / (name + '.log')}")
    meta = json.loads((raw / "scene_meta.json").read_text())
    meta["timing"]["stages"] = {k: v["wall_s"] for k, v in info["stages"].items()}
    meta["generator"] = dict(info, stages={k: dict(v, cmd=" ".join(v["cmd"])) for k, v in info["stages"].items()})
    (raw / "scene_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


# ------------------------------------------------------------------------------------------------ indoor (v2)
def _bbox(o):
    from infinigen.p4d.motion.objects import world_bbox

    return world_bbox(o)


def build_room(seed, T, W, H):
    import bpy
    import procfunc as pf

    from infinigen2.scenes.room import room, room_shape

    pf.ops.object.clear_scene()
    s = bpy.context.scene
    s.render.resolution_x, s.render.resolution_y = W, H
    s.render.fps = 24
    s.frame_start, s.frame_end = 1, T
    rng = np.random.default_rng(seed)
    rngs = rng.spawn(6)
    dims = room_shape.room_dimensions_rand(rngs[0])
    living = room.room_rand(rng=rngs[1], dimensions=dims, frame_start=1, frame_end=T)
    objs = [o.item() for o in living.all_objects]
    lights = [o.item() for o in living.lights]
    from infinigen2.util.scene_cleanup import cleanup_except

    cleanup_except(list(living.all_objects) + list(living.lights))
    return dict(objects=objs, lights=lights, floor=living.floor.item(), bbox=(np.zeros(3), np.array(dims, float)),
                living=living, rngs=rngs)


def build_flying(seed, T, W, H):
    """v2 flying_indoor scene (room + random-walk floating objects + lights), stereo camera removed."""
    import importlib.util

    import bpy

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("flying_indoor_render", root / "examples/flying_indoor/render.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    objects, lights, cams, times = mod.build_scene(seed, seed, 1, T, (W, H))
    for c in cams:
        bpy.data.objects.remove(c.item(), do_unlink=True)
    s = bpy.context.scene
    s.render.fps = 24
    objs = [o.item() for o in objects]
    for o in objs:
        if o.animation_data is not None and o.animation_data.action is not None:
            o["p4d_kind"], o["p4d_motion"] = "rigid", "random_walk"
    dims = np.max([_bbox(o)[1] for o in objs if "room_" in o.name], axis=0)
    return dict(objects=objs, lights=[l.item() for l in lights], bbox=(np.zeros(3), dims), times=times)


def indoor_cameras(rng, a, room, targets, moving_names):
    """4 cameras inside the room, independent path types, validated against the room BVH."""
    import bpy

    from infinigen.p4d import cameras as C

    s = bpy.context.scene
    T, fps = a.frames, 24.0
    clear_fn, ray_fn, bvh = C.blender_bvh_callbacks([o for o in room["objects"] if o.type == "MESH"])
    lo, hi = room["bbox"]
    m = 0.25

    def inside_fn(P):  # cameras must stay inside the room (walls are thin; clearance alone can't tell sides)
        P = np.asarray(P)
        return np.all((P >= lo + m) & (P <= hi - m), axis=-1)

    types = C.assign_path_types(rng, a.views, has_target=bool(targets))
    cams, metas = [], []
    tnames = list(targets)
    for v in range(a.views):
        cd = bpy.data.cameras.new(f"p4d_cam_{v}")
        cd.lens, cd.sensor_width, cd.sensor_fit = float(rng.uniform(14, 22)), 36.0, "HORIZONTAL"
        cd.clip_start, cd.clip_end = 0.05, 200
        cam = bpy.data.objects.new(f"p4d_cam_{v}", cd)
        s.collection.objects.link(cam)
        intr = C.Intrinsics.from_blender(cd.lens, cd.sensor_width, a.width, a.height)
        pt = types[v]
        tname = tnames[v % len(tnames)] if tnames else None
        target = targets[tname] if tname else None
        view = None
        for _ in range(40):  # anchor: inside the room, clear of geometry, 1.5-3.5 m from the target
            p = rng.uniform(lo + [0.4, 0.4, 0.8], hi - [0.4, 0.4, max(0.3, hi[2] - 2.1)])
            if clear_fn([p])[0] < 0.45:
                continue
            if target is not None:
                d = np.linalg.norm(p - target[0])
                dmax = 2.5 if a.family == "indoor_artic" else 4.0  # articulated parts are small: frame them closer
                if not (1.0 < d < dmax):
                    continue
            scale = float(np.clip(np.linalg.norm(p - target[0]) if target is not None else 2.5, 1.2, 4.0))
            view = C.sample_view(rng, T, fps, intr, p, scale, target=target, path_type=pt,
                                 jitter_prob=a.jitter_prob, clearance_fn=clear_fn, ray_fn=ray_fn, max_tries=8,
                                 inside_fn=inside_fn)
            if not view["meta"].get("fallback"):
                break
        C.blender_apply_path(cam, view["pos"], view["R"], frame_start=1)
        meta = dict(view["meta"], view_id=v, seed=int(a.seed * 100 + v), target_object=tname, lens_mm=cd.lens)
        cams.append(cam)
        metas.append(meta)
        logger.info(f"view {v}: {meta['path_type']} jitter={meta['jitter']} target={tname} "
                    f"attempts={meta['attempts']} fallback={meta.get('fallback', False)}")
    return cams, metas


def trajectories(names, T):
    import bpy

    s = bpy.context.scene
    out = {n: [] for n in names}
    for f in range(1, T + 1):
        s.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        for n in names:
            o = bpy.data.objects[n].evaluated_get(dg)
            bb = np.array([list(v) for v in o.bound_box])
            M = np.array(o.matrix_world)
            out[n].append(((bb @ M[:3, :3].T + M[:3, 3]).mean(0)))
    s.frame_set(1)
    return {n: np.array(v) for n, v in out.items()}


def run_indoor(a, out):
    import bpy

    from infinigen.p4d import gt
    from infinigen.p4d.motion import articulation, objects as P

    t0 = time.time()
    rng = np.random.default_rng(a.seed + 7)
    if a.family == "indoor_flying":
        room = build_flying(a.seed, a.frames, a.width, a.height)
    else:
        room = build_room(a.seed, a.frames, a.width, a.height)
    t_build = time.time() - t0
    s = bpy.context.scene
    s.cycles.film_exposure = 2.0
    motion = {}
    lo, hi = room["bbox"]
    t1 = time.time()
    movers = []
    if a.family == "indoor_physics":
        from mathutils.bvhtree import BVHTree  # noqa: F401

        from infinigen.p4d import cameras as C

        _, _, bvh = C.blender_bvh_callbacks([o for o in room["objects"] if o.type == "MESH"])
        small = []
        chairs = []
        for o in room["objects"]:
            if o.type != "MESH" or o.name.startswith("room_"):
                continue
            mn, mx = _bbox(o)
            if "chair" in o.name.lower():
                chairs.append(o)
            elif np.max(mx - mn) < 0.6:
                small.append(o)
        motion["physics"] = P.setup_physics(rng, room["objects"], small, room["bbox"], 1, a.frames, chairs=chairs,
                                            bvh=bvh)
        tb = time.time()
        P.bake(1, a.frames)
        motion["physics"]["bake_s"] = time.time() - tb
        movers = motion["physics"]["movers"]
        motion["physics"]["stats"] = P.motion_stats(movers, 1, a.frames)
    elif a.family == "indoor_artic":
        occupied = []
        for o in room["objects"]:
            if o.type != "MESH" or o.name.startswith("room_"):
                continue
            mn, mx = _bbox(o)
            if mx[2] - mn[2] < 0.05:  # rugs / flat decor: assets may stand on them
                continue
            occupied.append((mn, mx))
        arts, summ = articulation.add_articulated(rng, room["bbox"], occupied, 1, a.frames, n=(2, 3))
        motion["articulated"] = summ
        movers = [o.name for o in arts]
        room["objects"] += arts
    elif a.family == "indoor_flying":
        movers = [o.name for o in room["objects"] if o.get("p4d_motion") == "random_walk"]
        motion["flying"] = dict(n_random_walk=len(movers))
    t_motion = time.time() - t1
    # camera targets: moving objects (trajectory of bbox centre) + room centre
    targets = trajectories(movers, a.frames) if movers else {}
    if a.family == "indoor_physics" and targets:  # prefer objects that really move
        targets = {k: v for k, v in targets.items() if np.ptp(v, axis=0).max() > 0.05} or targets
    if not targets:
        targets = {"room_center": np.tile((lo + hi) / 2 * [1, 1, 0] + [0, 0, 0.9], (a.frames, 1))}
    t2 = time.time()
    cams, metas = indoor_cameras(rng, a, room, targets, movers)
    t_cams = time.time() - t2
    meta = gt.export_scene(out / raw_name(a), cams, a.family, a.seed, views_meta=metas, motion=motion,
                           samples=a.samples, extra_timing=dict(scene_build_s=t_build, motion_setup_s=t_motion,
                                                               cameras_s=t_cams))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--jitter_prob", type=float, default=0.25)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.family.startswith("nature"):
        meta = run_nature(a, a.out)
    else:
        meta = run_indoor(a, a.out)
    logger.info(f"DONE {a.family} seed {a.seed} in {time.time() - t0:.0f}s: {meta['tracks']['n']} tracks")


if __name__ == "__main__":
    main()
