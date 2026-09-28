"""Check rest triangulation on an actual populated creature scene before rendering."""
import argparse
import json
import os
import sys
from pathlib import Path

from infinigen.p4d.runtime import configure_cpu_budget
configure_cpu_budget()

import numpy as np


def run(blend, out):
    import bpy
    from infinigen.p4d.gt import (unhide_renderables, object_kind, eval_mesh,
                                  stabilize_triangulation, renderable)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    unhide_renderables()
    scene = bpy.context.scene
    objects = [o for o in scene.objects if o.type == "MESH"
               and object_kind(o) == "creature" and renderable(o)]
    modified = stabilize_triangulation(objects)
    initial, changed = {}, {}
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        for obj in objects:
            co, tri = eval_mesh(obj, dg)
            if co is None:
                raise ValueError(f"missing evaluated mesh: {obj.name}")
            if obj.name not in initial:
                initial[obj.name] = (len(co), tri)
            elif len(co) != initial[obj.name][0] or not np.array_equal(tri, initial[obj.name][1]):
                changed.setdefault(obj.name, frame)
    report = dict(PASS=bool(objects) and not changed, objects=len(objects),
                  frames=scene.frame_end - scene.frame_start + 1,
                  rest_triangulated=modified, changed=changed)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report["PASS"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blend", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    passed = run(args.blend, args.out)
    sys.stdout.flush()
    os._exit(0 if passed else 1)
