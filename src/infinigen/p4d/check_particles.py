"""Rendered regression for particle lifetimes and Cycles source-template IDs."""
import argparse
import json
from pathlib import Path

import numpy as np


def run(out):
    import bpy
    from infinigen.p4d import cameras as C, gt

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end, scene.render.fps = 1, 8, 24
    scene.render.resolution_x, scene.render.resolution_y = 128, 96
    bpy.ops.mesh.primitive_plane_add(size=10)
    bpy.context.object.name = "ground"
    bpy.ops.mesh.primitive_ico_sphere_add(radius=.15, location=(100, 100, 100))
    source = bpy.context.object
    source.name = "particle_source"
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 1.5))
    emitter = bpy.context.object
    emitter.name = "emitter"
    bpy.ops.object.particle_system_add()
    particles = emitter.particle_systems[-1].settings
    particles.type, particles.count = "EMITTER", 20
    particles.frame_start, particles.frame_end, particles.lifetime = 1, 3, 20
    particles.normal_factor, particles.particle_size = 0, 1
    particles.render_type, particles.instance_object = "OBJECT", source
    emitter.show_instancer_for_render = emitter.show_instancer_for_viewport = False
    bpy.ops.object.light_add(type="AREA", location=(0, -2, 5))
    bpy.context.object.data.energy = 400
    bpy.ops.object.camera_add(location=(4, -4, 3))
    camera = bpy.context.object
    camera.data.lens = 35
    positions = np.tile([4, -4, 3], (8, 1))
    C.blender_apply_path(camera, positions, C.look_rotation(np.tile([0, 0, 1], (8, 1)) - positions))
    meta = gt.export_scene(out, [camera], "particle_fixture", 0, n_mesh=0, n_inst=80, n_static=40, samples=8)
    tracks = np.load(out / "tracks_raw.npz")
    particle_tracks = tracks["source"] == 2
    assert particle_tracks.sum() == 80
    assert np.all(tracks["pass_index"][particle_tracks] == source.pass_index)
    alive = np.isfinite(tracks["xyz_world"][:, particle_tracks]).all(-1)
    assert np.any(~alive[0] & alive[-1]), "particles born after frame 1 were missed"
    index = gt._exr(out / "view_00" / "passes_0008.exr")["IndexOB.V"]
    assert np.any(np.round(index) == source.pass_index), "track IDs do not match rendered instances"
    assert meta["tracks"]["by_kind"]["particle"] == 80
    report = dict(PASS=True, particle_tracks=80, later_born_tracks=int((~alive[0] & alive[-1]).sum()),
                  instance_granularity="source_template")
    (out / "regression.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
