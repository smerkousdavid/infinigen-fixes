"""Small rendered calibration fixture; run before allocating long generation jobs."""
import argparse
import json
from pathlib import Path
import numpy as np


def run(out):
    import bpy
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    from infinigen.p4d import cameras as C, gt
    from infinigen.p4d.motion.creatures import rigid_residual

    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.resolution_x, s.render.resolution_y = 192, 128
    s.render.resolution_percentage, s.render.fps = 100, 24
    s.frame_start, s.frame_end = 1, 4
    bpy.ops.mesh.primitive_plane_add(size=30)
    ground = bpy.context.object
    ground.name = "floor"
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, .5))
    obj = bpy.context.object
    obj.name = "calibration_cube"
    obj["p4d_kind"] = "rigid"
    for f in range(1, 5):
        obj.location.x = .1 * (f - 1)
        obj.rotation_euler.z = .05 * (f - 1)
        obj.keyframe_insert("location", frame=f)
        obj.keyframe_insert("rotation_euler", frame=f)
    bpy.ops.object.light_add(type="AREA", location=(0, -2, 5))
    bpy.context.object.data.energy = 1000
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 5
    cams = []
    for i in range(4):
        bpy.ops.object.camera_add(location=(i * .5, -5, 2.5))
        cam = bpy.context.object
        cam.rotation_euler = (Vector((0, 0, .5)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
        cam.data.lens, cam.data.sensor_fit = 35, "HORIZONTAL"
        cams.append(cam)
    points = np.array([[0, 0, .5], [1, 0, 1], [-1, 1, 0], [2, 2, 2]])
    errors = []
    for fit, size, aspect, shift, percent in (
            ("HORIZONTAL", (192, 128), (1, 1), (0, 0), 100),
            ("VERTICAL", (128, 192), (1, 1), (.1, -.15), 50),
            ("AUTO", (128, 192), (2, 1), (-.12, .2), 100),
            ("AUTO", (192, 128), (1, 2), (.2, -.1), 75)):
        s.render.resolution_x, s.render.resolution_y = size
        s.render.pixel_aspect_x, s.render.pixel_aspect_y = aspect
        s.render.resolution_percentage = percent
        cam = cams[0]
        cam.data.sensor_fit, cam.data.shift_x, cam.data.shift_y = fit, *shift
        intr = C.Intrinsics.from_camera(cam, s, bpy.context.evaluated_depsgraph_get())
        K, E = gt.camera_matrices(cam, [1])
        pc = points @ E[0, :3, :3].T + E[0, :3, 3]
        uv = (pc @ K[0].T)[:, :2] / pc[:, 2:3]
        ref = np.array([world_to_camera_view(s, cam, Vector(p))[:] for p in points])
        ref = np.stack([ref[:, 0] * intr.width, (1 - ref[:, 1]) * intr.height], -1)
        error = float(np.max(np.abs(uv - ref)))
        assert error < 1e-3, (fit, size, aspect, shift, error)
        errors.append(error)
    s.render.resolution_x, s.render.resolution_y = 192, 128
    s.render.resolution_percentage = 100
    s.render.pixel_aspect_x = s.render.pixel_aspect_y = 1
    cams[0].data.sensor_fit, cams[0].data.shift_x, cams[0].data.shift_y = "HORIZONTAL", 0, 0
    x = np.random.default_rng(0).normal(size=(40, 3))
    rot = C.small_rotation([.5, .8, -.3])
    assert rigid_residual(x, x @ rot + [4, 1, 2]).max() < 1e-10
    changed = x.copy()
    changed[0, 0] += .5
    assert rigid_residual(x, changed).max() > .3
    # A transformed parent and animated lens must use evaluated state.
    parent = bpy.data.objects.new("camera_parent", None)
    s.collection.objects.link(parent)
    cam = cams[3]
    world = cam.matrix_world.copy()
    cam.parent = parent
    cam.matrix_world = world
    parent.location.x = .3
    parent.keyframe_insert("location", frame=1)
    parent.location.x = .6
    parent.keyframe_insert("location", frame=4)
    cam.data.lens = 35
    cam.data.keyframe_insert("lens", frame=1)
    cam.data.lens = 40
    cam.data.keyframe_insert("lens", frame=4)
    _, device, _ = gt.setup_render(samples=8)
    assert device == "GPU", "GPU preflight did not select a GPU"
    meta = gt.export_scene(out / "calibration", cams, "calibration", 0, samples=8,
                           n_mesh=200, n_static=400, n_inst=0,
                           views_meta=[dict(view_id=i, path_type="fixture", jitter=False) for i in range(4)])
    result = dict(camera_oracle_max_error_px=max(errors), device=meta["render"]["device"],
                  blender=meta["render"]["blender_version"], render_seconds=meta["timing"]["render_per_view_s"])
    (out / "checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.out)
