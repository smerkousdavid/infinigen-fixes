"""Compare batched population culling with native all-frame, all-camera culling."""
import json
import numpy as np


def run():
    import bpy
    from mathutils import Vector
    from infinigen.core.placement import placement
    from infinigen.p4d.visibility import patch_population_visibility

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 4
    scene.render.resolution_x, scene.render.resolution_y = 320, 180
    cameras, objects = [], []
    for i in range(4):
        bpy.ops.object.camera_add(location=(i - 1.5, -5, 2))
        camera = bpy.context.object
        camera.data.sensor_fit, camera.data.sensor_width, camera.data.sensor_height = "HORIZONTAL", 36, 20.25
        for frame in range(1, 5):
            camera.location.x = i - 1.5 + .1 * frame
            camera.rotation_euler = (Vector((0, 0, 1)) - camera.location).to_track_quat("-Z", "Y").to_euler()
            camera.keyframe_insert("location", frame=frame)
            camera.keyframe_insert("rotation_euler", frame=frame)
        cameras.append(camera)
    for i, position in enumerate(((0, 0, 1), (1, 0, 1), (100, 0, 1), (0, 5, 1))):
        bpy.ops.mesh.primitive_cube_add(size=1, location=position)
        obj = bpy.context.object
        obj.name = f"TreeFactory(1).spawn_placeholder({i})"
        objects.append(obj)
    scene.frame_set(1)
    expected = placement.filter_populate_targets(objects, cameras, 20., 1., False)
    patch_population_visibility()
    actual = placement.filter_populate_targets(objects, cameras, 20., 1., False)
    assert [o.name for o, _, _ in expected] == [o.name for o, _, _ in actual]
    assert np.allclose([r[1:] for r in expected], [r[1:] for r in actual], atol=1e-7)
    assert 0 < len(actual) < len(objects)
    print(json.dumps(dict(PASS=True, frames=4, cameras=4, placeholders=4, retained=len(actual))))


if __name__ == "__main__":
    from infinigen.p4d.runtime import configure_cpu_budget
    configure_cpu_budget()
    run()
