"""Blender integration checks for joint evaluation, wind and explicit impulses."""
import argparse
import json
from pathlib import Path
import numpy as np


def run(out):
    import bpy
    from infinigen.p4d.gt import eval_mesh
    from infinigen.p4d.motion import articulation as A, objects as P, wind

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end, scene.render.fps = 1, 48, 24
    report = {}
    for name in ("cabinet", "drawer"):
        obj = A.spawn_asset(name, 42)
        joints = A.animate_joints(np.random.default_rng(1), obj, 1, 48, profiles=["open"])
        samples = []
        for frame in (1, 24, 48):
            scene.frame_set(frame)
            co, tri = eval_mesh(obj, bpy.context.evaluated_depsgraph_get())
            samples.append((co, tri))
        assert all(np.array_equal(t, samples[0][1]) for _, t in samples)
        displacement = max(np.linalg.norm(co - samples[0][0], axis=1).max() for co, _ in samples)
        assert displacement > .05, (name, displacement)
        report[name] = dict(max_vertex_displacement_m=float(displacement),
                            joints=[{k: j[k] for k in ("kind", "min", "max", "units")} for j in joints])
        bpy.data.objects.remove(obj, do_unlink=True)

    scene.frame_set(1)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=1)
    plant = bpy.context.object
    plant.rotation_euler.y = np.pi / 2
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    wind.apply_wind([plant], strength=.12, seed=1)
    samples = []
    for frame in (1, 12, 24, 48):
        scene.frame_set(frame)
        samples.append(eval_mesh(plant, bpy.context.evaluated_depsgraph_get()))
    displacement = max(np.linalg.norm(c - samples[0][0], axis=1).max() for c, _ in samples)
    assert displacement > .01
    assert all(np.array_equal(t, samples[0][1]) for _, t in samples)
    report["wind"] = dict(max_vertex_displacement_m=float(displacement), topology_constant=True)
    bpy.data.objects.remove(plant, do_unlink=True)

    scene.frame_set(1)
    P._ensure_world(1, 48)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -.1), scale=(10, 10, .2))
    P._add_rb(bpy.context.object, "PASSIVE", "BOX")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=.1, location=(0, 0, .102))
    ball = bpy.context.object
    P._add_rb(ball, "ACTIVE", "SPHERE", mass=.2, friction=.5)
    summary = dict(movers=[ball.name], drops=[], slides=[], chairs=[], rolls=[dict(
        object=ball.name, radius_m=.1, v=[1., 0, 0], w=[0, 10., 0])])
    P.bake_explicit(1, 48, summary)
    positions = []
    for frame in range(1, 49):
        scene.frame_set(frame)
        positions.append(list(ball.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.translation))
    positions = np.asarray(positions)
    assert .65 < positions[23, 0] < 1.05
    assert np.max(np.abs(positions[:, 2] - .1)) < .005
    report["rolling"] = dict(displacement_2s_m=float(positions[-1, 0]),
                              max_contact_height_error_m=float(np.max(np.abs(positions[:, 2] - .1))),
                              simulation=summary["simulation"])
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
