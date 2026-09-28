"""Signed terrain-height checks and explicit repairs for prepared creature rigs."""
import json
import numpy as np


def reaim_creature_high(compact=False, seed=0):
    """Keep prepared trajectories, aiming all high views at evaluated actors.

    A formerly fixed-aim near-static view becomes a tracking pan. Original
    jitter is retained as a local rotation relative to the newly authored aim.
    The compact option authors close but distinct pan/crane/dolly/handheld
    trajectories. Geometry, actor motion, medium and low cameras are untouched.
    """
    import bpy
    from infinigen.p4d.motion.creatures import actor_bounds
    scene = bpy.context.scene
    frames = list(range(scene.frame_start, scene.frame_end + 1))
    roots = [o for o in scene.objects if o.get('p4d_gait_report')]
    bounds = actor_bounds(roots, frames)
    target = np.mean([value.mean(1) for value in bounds.values()], axis=0)
    return reaim_high_cameras(target, compact=compact, seed=seed,
                              aim='evaluated creature group centre')


def reaim_high_cameras(target, compact=False, seed=0, aim='shared scene target'):
    """Re-author only the high rig around an explicit per-frame world target."""
    import bpy
    from infinigen.core.placement.camera import get_camera_rigs
    from infinigen.p4d import cameras as C
    scene = bpy.context.scene
    frames = list(range(scene.frame_start, scene.frame_end + 1))
    target = C._target_at(target, len(frames))
    rigs = get_camera_rigs()
    selected = [r for r in rigs if json.loads(r['p4d_view']).get('overlap_target') == 'high']
    if len(selected) != 4:
        raise ValueError('requires four high-overlap cameras')
    matrices = []
    for frame in frames:
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        matrices.append([np.asarray(next(c for c in r.children if c.type == 'CAMERA').evaluated_get(dg).matrix_world)
                         for r in selected])
    matrices = np.asarray(matrices)
    from infinigen.p4d.motion.creatures import terrain_bvh
    ground = terrain_bvh(render_only=True) if compact else None
    records = []
    for i, rig in enumerate(selected):
        cam = next(c for c in rig.children if c.type == 'CAMERA')
        if rig.parent is not None or not np.allclose(cam.matrix_local, np.eye(4), atol=1e-6):
            raise ValueError('re-aim requires an unparented rig and identity camera child')
        meta = json.loads(rig['p4d_view'])
        pos, old_R = matrices[:, i, :3, 3], matrices[:, i, :3, :3]
        jitter = meta.get('jitter_params') if meta.get('jitter') else None
        base_pos = np.asarray(jitter['pre_jitter_pos']) if jitter else pos.copy()
        base_R = C.look_rotation(target - base_pos)
        previous_type = meta['path_type']
        if compact:
            path_type = ('spin', 'crane', 'dolly', 'handheld')[i]
            path_seed = [int(seed), i, 193]
            rng = np.random.default_rng(np.random.SeedSequence(path_seed))
            anchor = matrices[0, 0, :3, 3] + rng.normal(0, .015, 3)
            scale = float(np.linalg.norm(target[0] - anchor))
            base_pos, base_R, params = C.PATHS[path_type](rng, len(frames), anchor, target, scale,
                fps=scene.render.fps / scene.render.fps_base, track_target=np.ptp(target, axis=0).max() > .01,
                yaw_sweep_deg=2., pitch_amp_deg=.1,
                dz=.08, drift_scale=.002, frac=.01, side_scale=.001, speed=.01, look_noise_deg=.1)
            meta.update(path_type=path_type, params=params, compact_high_seed=path_seed)
        if jitter:
            delta = pos - np.asarray(jitter['pre_jitter_pos'])
            noise = np.swapaxes(np.asarray(jitter['pre_jitter_R_cw']), 1, 2) @ old_R
            pos = base_pos + delta
            R = base_R @ noise
            jitter['pre_jitter_pos'] = base_pos.tolist()
            jitter['pre_jitter_R_cw'] = base_R.tolist()
        else:
            pos, R = base_pos, base_R
        if compact and (ground is None or heights_above_terrain(pos, ground).min() < .3):
            raise ValueError('compact high path has inadequate terrain clearance')
        C.blender_apply_path(rig, pos, R, frame_start=frames[0])
        record = dict(camera=cam.name, previous_path_type=previous_type,
                      aim=aim, translation_unchanged=not compact,
                      compact=compact)
        if meta['path_type'] == 'static':
            meta['path_type'] = 'spin'
        meta['params']['aim'] = aim
        meta['high_reaim'] = record
        meta['target_world'] = target.tolist()
        rig['p4d_view'] = json.dumps(meta)
        records.append(record)
    scene['p4d_views'] = json.dumps([json.loads(r['p4d_view']) for r in rigs])
    return dict(frames_checked=len(frames), cameras=records)


def heights_above_terrain(points, bvh):
    from mathutils import Vector
    result = []
    for point in points:
        hit = bvh.ray_cast(Vector((point[0], point[1], 1e4)), Vector((0, 0, -1)))
        if hit[0] is None:
            raise ValueError("camera has no terrain below its position")
        result.append(float(point[2] - hit[0].z))
    return np.asarray(result)


def repair_creature_cameras():
    """Lift buried paths by one constant offset, keeping their motion and jitter.

    Only invalid paths change. Re-aiming uses evaluated actor group centres;
    the original local rotational jitter is applied to the new aim trajectory.
    """
    import bpy
    from infinigen.core.placement.camera import get_camera_rigs
    from infinigen.p4d import cameras as C
    from infinigen.p4d.motion.creatures import actor_bounds, terrain_bvh
    scene = bpy.context.scene
    frames = list(range(scene.frame_start, scene.frame_end + 1))
    roots = [o for o in scene.objects if o.get("p4d_gait_report")]
    bounds = actor_bounds(roots, frames)
    target = np.mean([value.mean(1) for value in bounds.values()], axis=0)
    rigs = get_camera_rigs()
    cams = [next(c for c in r.children if c.type == 'CAMERA') for r in rigs]
    bvh = terrain_bvh(render_only=True)
    if bvh is None:
        raise ValueError("missing final terrain")
    positions, rotations, heights = [], [], []
    for frame in frames:
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        matrices = np.asarray([c.evaluated_get(dg).matrix_world for c in cams])
        positions.append(matrices[:, :3, 3])
        rotations.append(matrices[:, :3, :3])
        heights.append(heights_above_terrain(matrices[:, :3, 3], bvh))
    positions, rotations, heights = map(np.asarray, (positions, rotations, heights))
    records = []
    for i, (rig, cam) in enumerate(zip(rigs, cams)):
        before = float(heights[:, i].min())
        if before >= .3:
            continue
        if not np.allclose(cam.matrix_local, np.eye(4), atol=1e-6) or rig.parent is not None:
            raise ValueError("terrain repair requires an unparented rig and identity camera child")
        lift = 1.7 - before
        pos = positions[:, i].copy()
        pos[:, 2] += lift
        meta = json.loads(rig['p4d_view'])
        jitter = meta.get('jitter_params') if meta.get('jitter') else None
        if jitter:
            base_pos = np.asarray(jitter['pre_jitter_pos']).copy()
            base_pos[:, 2] += lift
            noise = np.swapaxes(np.asarray(jitter['pre_jitter_R_cw']), 1, 2) @ rotations[:, i]
            base_R = C.look_rotation(target - base_pos)
            R = base_R @ noise
            jitter.update(pre_jitter_pos=base_pos.tolist(), pre_jitter_R_cw=base_R.tolist())
        else:
            R = C.look_rotation(target - pos)
        C.blender_apply_path(rig, pos, R, frame_start=frames[0])
        record = dict(camera=cam.name, original_min_height_m=before, translation_z_m=lift,
                      minimum_height_m=before + lift, aim='evaluated creature group centre')
        meta['terrain_repair'] = record
        rig['p4d_view'] = json.dumps(meta)
        heights[:, i] += lift
        records.append(record)
    scene['p4d_views'] = json.dumps([json.loads(r['p4d_view']) for r in rigs])
    return dict(frames_checked=len(frames), min_per_view_m=heights.min(0).tolist(), repairs=records)
