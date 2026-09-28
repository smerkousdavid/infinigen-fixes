"""Sparse rendered-overlap preflight for prepared nature scenes.

This diagnostic uses the final resolution and the production Position/Object
Index passes. It never replaces acceptance measurements across the full clip.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from infinigen.p4d.runtime import configure_cpu_budget, source_identity
configure_cpu_budget()


def run(a):
    import bpy
    import OpenEXR
    from infinigen.core.placement.camera import get_camera_rigs
    from infinigen.p4d import gt
    from infinigen.p4d.motion import wind
    from infinigen.p4d.motion.creatures import prepare_scene_contacts
    sys.path.insert(0, str(a.p4d_repo / 'stream'))
    from p4d_data.camera_overlap import measure_camera_overlap

    config = json.loads((a.source / 'resolved_config.json').read_text())
    stages = json.loads((a.source / 'run.json').read_text())
    blend = a.source / 'fine/scene.blend'
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.context.scene
    gt.unhide_renderables()
    prepare_scene_contacts()
    unique, sources = wind.vegetation_targets(scene)
    wind.apply_wind(unique + sources, strength=stages['wind_strength'],
                    gust=.6, flutter=stages['flutter'], seed=config['seed'])
    gt.stabilize_triangulation(bpy.data.objects)
    # Include hidden source templates: their instances can be visible.
    gt.assign_pass_indices([o for o in bpy.data.objects if o.type in
                            ('MESH', 'CURVE', 'CURVES', 'FONT', 'META', 'VOLUME', 'POINTCLOUD')])
    fo, device, _ = gt.setup_render(samples=a.samples, width=config['width'], height=config['height'])
    if device != 'GPU':
        raise RuntimeError('rendered-overlap preflight requires GPU')
    scene.cycles.seed = config['seed']
    scene.cycles.use_animated_seed = False
    if any(f < scene.frame_start or f > scene.frame_end for f in a.frames):
        raise ValueError('diagnostic frame outside prepared animation')
    # Production tracking evaluates the entire timeline before camera rendering.
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
    rigs = get_camera_rigs()
    result = dict(source=str(a.source), blend_sha256=hashlib.sha256(blend.read_bytes()).hexdigest(),
                  source_identity=source_identity(), frames=a.frames, samples=a.samples,
                  diagnostic_only=True, tiers={})
    for tier in a.tiers:
        selected = [r for r in rigs if json.loads(r['p4d_view']).get('overlap_target') == tier]
        if len(selected) != 4:
            raise ValueError(f'{tier}: expected four prepared cameras')
        Ks, Es, depths, indices = [], [], [], []
        for v, rig in enumerate(selected):
            cam = next(c for c in rig.children if c.type == 'CAMERA')
            cam.data.dof.use_dof = False
            scene.camera = cam
            folder = a.out / tier / f'view_{v:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            fo.base_path = str(folder / 'passes_')
            k, e, dep, ids = [], [], [], []
            for frame in a.frames:
                scene.frame_set(frame)
                K, E = gt.evaluated_camera_matrices(cam, scene, bpy.context.evaluated_depsgraph_get())
                scene.render.filepath = str(folder / f'rgb_{frame:04d}.png')
                bpy.ops.render.render(write_still=True)
                with OpenEXR.File(str(folder / f'passes_{frame:04d}.exr'), separate_channels=True) as f:
                    ch = {name: np.asarray(value.pixels) for name, value in f.channels().items()}
                position = np.stack([ch[f'Position.{c}'] for c in 'XYZ'], -1)
                z = (position @ E[:3, :3].T + E[:3, 3])[..., 2]
                valid = np.isfinite(z) & (z > 0) & (ch['Depth.V'] > 0) & (ch['Depth.V'] < 1e5)
                dep.append(np.where(valid, z, 0))
                ids.append(np.round(ch['IndexOB.V']).astype(np.int32) * valid)
                k.append(K); e.append(E)
            Ks.append(k); Es.append(e); depths.append(dep); indices.append(ids)
        arrays, summary = measure_camera_overlap(np.asarray(Ks), np.asarray(Es),
            lambda v, t: depths[v][t], lambda v, t: indices[v][t], requested=tier)
        np.savez_compressed(a.out / tier / 'overlap.npz', **arrays)
        result['tiers'][tier] = summary
        (a.out / 'overlap.json').write_text(json.dumps(result, indent=2))
        print('SPARSE_OVERLAP', tier, summary['score'], summary['achieved'], flush=True)
    result['PASS'] = all(v['requested'] == v['achieved'] and v['valid_pair_frame_fraction'] == 1.
                         for v in result['tiers'].values())
    (a.out / 'overlap.json').write_text(json.dumps(result, indent=2))
    return result['PASS']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--p4d-repo', type=Path, default=Path('/root/point4d-datasets'))
    p.add_argument('--tiers', nargs='+', choices=('high', 'medium', 'low'), default=['high', 'medium', 'low'])
    p.add_argument('--frames', nargs='+', type=int, default=[1, 48, 96])
    p.add_argument('--samples', type=int, default=16)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    passed = run(args)
    sys.stdout.flush()
    os._exit(0 if passed else 1)
