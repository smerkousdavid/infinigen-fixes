"""Reframe a prepared wind scene around its actual populated vegetation."""
import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from infinigen.p4d.runtime import configure_cpu_budget, source_identity
configure_cpu_budget()


def run(source, out):
    import bpy
    from mathutils import Vector
    from infinigen.core.placement.camera import get_camera_rigs
    from infinigen.p4d import cameras as C, gt
    from infinigen.p4d.rigs import sample_rig, TIERS
    from infinigen.p4d.motion.creatures import terrain_bvh
    start = time.time()
    config = json.loads((source / 'resolved_config.json').read_text())
    stages = json.loads((source / 'run.json').read_text())
    if config['family'] != 'nature_wind' or any(stages['stages'][stage]['exit'] != 0
            for stage in ('coarse', 'populate', 'fine_terrain')):
        raise ValueError('requires successful wind geometry stages')
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('coarse', 'populated', 'fine'):
        shutil.copytree(source / folder, out / folder)
    (out / 'resolved_config.json').write_text(json.dumps(config, indent=2))
    blend = out / 'fine' / 'scene.blend'
    before = hashlib.sha256(blend.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    gt.unhide_renderables()
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    dg = bpy.context.evaluated_depsgraph_get()
    trees = [o for o in scene.objects if 'TreeFactory' in o.name and 'spawn_asset' in o.name
             and o.type == 'MESH' and gt.renderable(o)]
    centres = []
    for tree in trees:
        evaluated = tree.evaluated_get(dg)
        matrix = np.asarray(evaluated.matrix_world)
        bounds = np.asarray(evaluated.bound_box) @ matrix[:3, :3].T + matrix[:3, 3]
        centres.append((bounds.min(0) + bounds.max(0)) / 2)
    if len(centres) < 2:
        raise ValueError('wind camera preparation requires populated trees')
    centres = np.asarray(centres)
    # Use a real tree near the cluster median rather than a point that could lie
    # in open water between clusters. Neighbouring trees provide low-tier targets.
    order = np.argsort(np.linalg.norm(centres[:, :2] - np.median(centres[:, :2], axis=0), axis=1))
    centre = centres[order[0]].copy()
    ground = terrain_bvh(render_only=True)
    def ground_z(x, y):
        hit = ground.ray_cast(Vector((x, y, 1e4)), Vector((0, 0, -1)))
        return None if hit[0] is None else hit[0].z
    centre[2] = ground_z(*centre[:2]) + 3.
    renderables = [o for o in scene.objects if o.type == 'MESH' and gt.renderable(o)
                  and not len(o.particle_systems)]
    clear, rays, _ = C.blender_bvh_callbacks(renderables)
    T = scene.frame_end - scene.frame_start + 1
    fps = scene.render.fps / scene.render.fps_base
    target = np.tile(centre, (T, 1))
    def inside(points):
        heights = [ground_z(*p[:2]) for p in points]
        return np.asarray([z is not None and p[2] - z >= .5 for p, z in zip(points, heights)])
    def anchor(rng):
        for _ in range(300):
            angle = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(10, 18)
            xy = centre[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
            z = ground_z(*xy)
            if z is None:
                continue
            point = np.r_[xy, z + rng.uniform(1.6, 2.5)]
            if clear([point])[0] > .6:
                return point
        raise ValueError('no clear vegetation camera anchor')
    rigs = get_camera_rigs()
    if len(rigs) != 12:
        raise ValueError('prepared scene must have all twelve cameras')
    records, authored = [], []
    for k, tier in enumerate(TIERS):
        group = rigs[4*k:4*k+4]
        intrinsics = []
        for rig in group:
            cam = next(c for c in rig.children if c.type == 'CAMERA')
            cam.data.lens = 24.
            cam.data.sensor_width, cam.data.sensor_fit = 36., 'HORIZONTAL'
            cam.data.sensor_height = 36. * scene.render.resolution_y / scene.render.resolution_x
            intrinsics.append(C.Intrinsics.from_camera(cam, scene, bpy.context.evaluated_depsgraph_get()))
        targets = [target] * 4
        if tier == 'low':
            # Separated real tree regions, all with nonrigid vegetation evidence.
            candidates = centres[np.argsort(centres[:, 0])]
            selected = candidates[np.linspace(0, len(candidates)-1, 4).astype(int)]
            targets = [np.tile(p, (T, 1)) for p in selected]
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], k, 917]))
        sampled = sample_rig(rng, T, fps, intrinsics, anchor, target, tier, clear, rays,
                             inside_fn=inside, view_targets=targets, max_tries=400)
        for i, (rig, view) in enumerate(zip(group, sampled)):
            C.blender_apply_path(rig, view['pos'], view['R'], frame_start=scene.frame_start)
            meta = dict(view['meta'], view_id=i, seed=int(config['seed']*100+k*4+i),
                        target_object='populated_vegetation', lens_mm=24.,
                        target_world=targets[i].tolist())
            rig['p4d_view'] = json.dumps(meta)
            records.append(meta)
            authored.append((rig.name, view['pos'], view['R'], meta))
        print(json.dumps(dict(tier=tier, preview_overlap=sampled[0]['meta']['preview_overlap'])), flush=True)
    scene['p4d_views'] = json.dumps(records)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    report = dict(source=str(source), input_sha256=before,
                  output_sha256=hashlib.sha256(blend.read_bytes()).hexdigest(),
                  source_identity=source_identity(), seconds=time.time()-start,
                  target_tree=trees[order[0]].name, target_world=centre.tolist(), views=records)
    # Fine terrain is camera-dependent. Update upstream scene rigs and require
    # remeshing before any final render uses this camera configuration.
    for folder, stage in (('coarse', 'coarse'), ('populated', 'populate')):
        path = out / folder / 'scene.blend'
        original = hashlib.sha256(path.read_bytes()).hexdigest()
        bpy.ops.wm.open_mainfile(filepath=str(path))
        current = bpy.context.scene
        for name, pos, R, meta in authored:
            rig = bpy.data.objects[name]
            C.blender_apply_path(rig, pos, R, frame_start=current.frame_start)
            rig['p4d_view'] = json.dumps(meta)
            cam = next(c for c in rig.children if c.type == 'CAMERA')
            cam.data.lens, cam.data.sensor_width, cam.data.sensor_fit = 24., 36., 'HORIZONTAL'
            cam.data.sensor_height = 36. * current.render.resolution_y / current.render.resolution_x
        current['p4d_views'] = json.dumps(records)
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        stages['stages'][stage]['camera_postprocess'] = dict(
            input_sha256=original, output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_identity=source_identity(), prepared_camera_source=report)
    stages['stages']['fine_terrain'] = dict(exit=None, status='requires_camera_dependent_remeshing',
        source_stage=stages['stages']['fine_terrain'], camera_postprocess=report)
    report['seconds'] = time.time() - start
    (out / 'run.json').write_text(json.dumps(stages, indent=2))
    (out / 'preparation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(seconds=report['seconds'], target=report['target_tree'])), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
    sys.stdout.flush()
    os._exit(0)
