"""Compare filtered and exhaustive clearance on a generated nature scene."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from infinigen.p4d.runtime import configure_cpu_budget
configure_cpu_budget()
import numpy as np


def run(blend, out):
    import bpy
    from infinigen.p4d import gt
    from infinigen.p4d.motion import wind
    from infinigen.core.placement.camera import get_camera_rigs
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    gt.unhide_renderables()
    unique, sources = wind.vegetation_targets()
    wind.apply_wind(unique + sources, strength=.058, flutter=.022, gust=.6, seed=102)
    gt.stabilize_triangulation(bpy.data.objects)
    cameras = [next(c for c in r.children if c.type == "CAMERA") for r in get_camera_rigs()]
    objects = [o for o in bpy.context.scene.objects if o.type == "MESH" and gt.renderable(o)]
    frames = sorted({bpy.context.scene.frame_start, (bpy.context.scene.frame_start + bpy.context.scene.frame_end)//2,
                     bpy.context.scene.frame_end})
    reports, timing = {}, {}
    for name, enabled in (("full", False), ("filtered", True)):
        start = time.time()
        reports[name] = gt.clearance_report(objects, cameras, frames, broadphase=enabled)
        timing[name] = time.time() - start
    difference = float(np.max(np.abs(np.array(reports['full']['per_frame_m']) - reports['filtered']['per_frame_m'])))
    assert difference < 1e-5, difference
    report = dict(PASS=True, max_difference_m=difference, timing_s=timing, cameras=len(cameras), frames=frames,
                  moving_objects=reports['filtered']['moving_objects'],
                  candidates=reports['filtered']['candidate_objects_per_frame'])
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blend", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.blend, args.out)
    sys.stdout.flush()
    os._exit(0)
