# Copyright (C) 2026. This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.
"""Batch runner for p4d scenes on one pod (Stage B).

    python -m infinigen.p4d.batch --plan plan.txt --out /root/out/batch --jobs 3 [--p4d_repo /root/point4d-datasets]
                                  [--frames 48 --width 640 --height 360 --views 4 --samples 128]

plan.txt: one "family seed" per line. Up to --jobs scenes run concurrently (the v1 nature stages are CPU bound and
single threaded, so several scenes overlap well; renders share the GPU). After each scene:
  1. QA: the p4d-1.1 converter + validator from point4d-datasets (convert/infinigen.py) must pass;
  2. the raw scene dir is packed to <out>/ready/<scene>.tar for transfer, and v1 intermediates (coarse/fine/frames,
     several GB) are deleted.
Progress/timings: <out>/batch.jsonl (one record per scene). Re-running skips scenes already in ready/ or failed/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_one(fam, seed, a):
    name = f"{fam}_s{seed:05d}"
    work = a.out / "work" / name
    ready = a.out / "ready" / f"{name}.tar"
    if ready.exists() or (a.out / "failed" / f"{name}.json").exists():
        return None
    work.mkdir(parents=True, exist_ok=True)
    rec = dict(scene=name, family=fam, seed=seed, start=time.time())
    cmd = [sys.executable, "-m", "infinigen.p4d.run_scene", "--family", fam, "--seed", str(seed), "--out", str(work),
           "--frames", str(a.frames), "--width", str(a.width), "--height", str(a.height), "--views", str(a.views),
           "--samples", str(a.samples)]
    t0 = time.time()
    with open(work / "run_scene.log", "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(a.src))
    rec["generate_s"] = round(time.time() - t0, 1)
    raw = work / name
    if r.returncode != 0 or not (raw / "scene_meta.json").exists():
        rec.update(status="generate_failed", exit=r.returncode,
                   log_tail=(work / "run_scene.log").read_text(errors="ignore")[-3000:])
        return _fail(a, name, rec)
    # QA conversion (validator) on the pod
    t1 = time.time()
    qa = a.out / "qa"
    qa.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(a.p4d_repo / "stream"))
    q = subprocess.run([sys.executable, str(a.p4d_repo / "convert/infinigen.py"), str(raw), "--out", str(qa)],
                       capture_output=True, text=True, env=env)
    rec["qa_s"] = round(time.time() - t1, 1)
    qjson = next(iter(qa.glob(f"*{name}.quality.json")), None)
    if q.returncode != 0 or qjson is None:
        rec.update(status="qa_failed", qa_tail=(q.stdout + q.stderr)[-3000:])
        return _fail(a, name, rec)
    qual = json.loads(qjson.read_text())["quality"]
    rec["quality"] = {k: qual.get(k) for k in ("reprojection_agree_mean", "reprojection_agree_min_view",
                                                "cross_view_agree", "n_tracks", "tracks_per_kind", "flow_convention",
                                                "flow_dropped", "validated")}
    for t in qa.glob(f"*{name}.tar"):  # QA clip not needed on the pod (the pipeline rebuilds it from raw)
        t.unlink()
    meta = json.loads((raw / "scene_meta.json").read_text())
    rec["timing"] = meta.get("timing")
    rec["tracks"] = {k: meta["tracks"][k] for k in ("n", "by_source", "by_kind")}
    rec["views"] = [dict(path_type=v.get("path_type"), jitter=v.get("jitter"), fallback=v.get("fallback", False))
                    for v in meta.get("views", [])]
    rec["motion_keys"] = sorted(meta.get("motion", {}))
    t2 = time.time()
    (a.out / "ready").mkdir(exist_ok=True)
    with tarfile.open(str(ready) + ".partial", "w") as tar:
        tar.add(raw, arcname=name)
    os.replace(str(ready) + ".partial", ready)
    rec["pack_s"] = round(time.time() - t2, 1)
    rec["ready_bytes"] = ready.stat().st_size
    shutil.rmtree(work, ignore_errors=True)
    rec.update(status="ok", end=time.time(), wall_s=round(time.time() - rec["start"], 1))
    _log(a, rec)
    return rec


def _fail(a, name, rec):
    (a.out / "failed").mkdir(exist_ok=True)
    rec["end"] = time.time()
    (a.out / "failed" / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str))
    _log(a, rec)
    return rec


def _log(a, rec):
    with open(a.out / "batch.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("scene", "status", "wall_s", "generate_s")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--p4d_repo", type=Path, default=Path("/root/point4d-datasets"))
    ap.add_argument("--src", type=Path, default=Path("/root/infinigen"))
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--samples", type=int, default=128)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plan = [ln.split() for ln in a.plan.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda fs: run_one(fs[0], int(fs[1]), a), plan))
    print("BATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
