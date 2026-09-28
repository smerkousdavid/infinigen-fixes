"""Checkpoint validated creature motion into a copied fine-stage scene."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from infinigen.p4d.runtime import configure_cpu_budget, source_identity
configure_cpu_budget()


def run(source, out, reaim_high=False, compact_high=False):
    import bpy
    from infinigen.p4d import gt
    from infinigen.p4d.motion.creatures import prepare_scene_contacts
    start = time.time()
    config = json.loads((source / "resolved_config.json").read_text())
    stages = json.loads((source / "run.json").read_text())
    if config["family"] != "nature_creatures" or any(stages["stages"][stage]["exit"] != 0
            for stage in ("coarse", "populate", "fine_terrain")):
        raise ValueError("requires successful creature geometry stages")
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("coarse", "populated", "fine"):
        shutil.copytree(source / folder, out / folder)
    (out / "resolved_config.json").write_text(json.dumps(config, indent=2))
    blend = out / "fine" / "scene.blend"
    before = hashlib.sha256(blend.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    gt.unhide_renderables()
    gt.stabilize_triangulation(bpy.data.objects)
    report = prepare_scene_contacts()
    from infinigen.p4d.terrain_cameras import repair_creature_cameras
    report['camera_terrain'] = repair_creature_cameras()
    if reaim_high or compact_high:
        from infinigen.p4d.terrain_cameras import reaim_creature_high
        report['camera_high_reaim'] = reaim_creature_high(compact=compact_high)
    report["gaits"] = {obj.name: json.loads(obj["p4d_gait_report"]) for obj in bpy.context.scene.objects
                       if obj.get("p4d_gait_report")}
    bpy.context.scene["p4d_motion_preparation"] = json.dumps(source_identity())
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    report.update(source=str(source), input_sha256=before, output_sha256=hashlib.sha256(blend.read_bytes()).hexdigest(),
                  source_identity=source_identity(), seconds=time.time()-start)
    if stages['stages']['fine_terrain'].get('motion_postprocess'):
        report['previous_motion_postprocess'] = stages['stages']['fine_terrain']['motion_postprocess']
    stages["stages"]["fine_terrain"]["motion_postprocess"] = report
    (out / "run.json").write_text(json.dumps(stages, indent=2))
    (out / "preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("actor_separation", "actor_clearance", "camera_terrain", "seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument('--reaim-high', action='store_true')
    parser.add_argument('--compact-high', action='store_true')
    args = parser.parse_args()
    run(args.source, args.out, args.reaim_high, args.compact_high)
    sys.stdout.flush()
    os._exit(0)
