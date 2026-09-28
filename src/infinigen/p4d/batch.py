# Copyright (C) 2026. This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.
"""Batch runner for p4d scenes on one pod (Stage B).

    python -m infinigen.p4d.batch --plan plan.txt --out /root/out/batch --jobs 3 [--p4d_repo /root/point4d-datasets]
                                  [--frames 48 --width 640 --height 360 --views 4 --samples 128]

plan.txt: one "family seed" per line. Up to --jobs scenes run concurrently (the v1 nature stages are CPU bound and
single threaded, so several scenes overlap well; renders share the GPU). After each scene:
  1. QA: the p4d-1.1 converter + validator from point4d-datasets (convert/infinigen.py) must pass;
  2. retain validated compressed clips in ready/, with SHA256 manifests; preserve raw until transfer verification.
Progress/timings: <out>/batch.jsonl (one record per scene). Re-running verifies completed clips and retries failed scenes.
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
    import hashlib
    from infinigen.p4d.qa import check

    name = f"{fam}_s{seed:05d}"
    work = a.out / "work" / name
    ready = a.out / "ready"
    ready.mkdir(parents=True, exist_ok=True)
    complete = ready / f"{name}.complete.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        if all((ready / c["file"]).exists() and hashlib.sha256((ready / c["file"]).read_bytes()).hexdigest() == c["sha256"]
               for c in previous["clips"]):
            return previous
        raise ValueError(f"completed scene has missing/corrupted converted clips: {name}")
    work.mkdir(parents=True, exist_ok=True)
    rec = dict(scene=name, family=fam, seed=seed, start=time.time(), clips=[])
    cmd = [sys.executable, "-m", "infinigen.p4d.run_scene", "--family", fam, "--seed", str(seed), "--out", str(work),
           "--frames", str(a.frames), "--width", str(a.width), "--height", str(a.height), "--views", str(a.views),
           "--fps", str(a.fps), "--samples", str(a.samples), "--overlap", a.overlap]
    if a.resume and (work / "resolved_config.json").exists():
        cmd.append("--resume")
    exit_code = 0
    if not a.collect_only:
        with open(work / "run_scene.log", "a") as log:
            exit_code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(a.src)).returncode
    rec["generate_s"] = round(time.time() - rec["start"], 1)
    raws = sorted(p.parent for p in work.glob(name + "*/scene_meta.json"))
    if exit_code != 0 or len(raws) != (3 if a.overlap == "all" else 1):
        log_path = work / "run_scene.log"
        rec.update(status="generate_failed", exit=exit_code,
                   log_tail=log_path.read_text(errors="ignore")[-3000:] if log_path.exists() else "raw clips incomplete")
        return _fail(a, name, rec)
    qa = a.out / "qa"
    qa.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(a.p4d_repo / "stream"))
    for raw in raws:
        meta = json.loads((raw / "scene_meta.json").read_text())
        if not meta.get("resolved_config") or meta.get("scene_id") != name:
            rec.update(status="generate_incomplete", raw=str(raw),
                       reason="scene driver has not finalized configuration/provenance")
            return _fail(a, name, rec)
        meta["split"] = a.split
        (raw / "scene_meta.json").write_text(json.dumps(meta, indent=1))
        report = qa / f"{raw.name}.conversion.json"
        q = subprocess.run([sys.executable, str(a.p4d_repo / "convert/infinigen.py"), str(raw), "--out", str(qa),
                            "--report", str(report)],
                           capture_output=True, text=True, env=env)
        (qa / f"{raw.name}.conversion.log").write_text(q.stdout + q.stderr)
        clips = list(qa.glob(f"*__{raw.name}.tar"))
        if q.returncode != 0 or len(clips) != 1:
            rec.update(status="qa_failed", qa_tail=(q.stdout + q.stderr)[-3000:])
            return _fail(a, name, rec)
        gates = check(raw, clips[0], expected_frames=a.frames, expected_size=(a.width, a.height), expected_fps=a.fps)
        (qa / f"{raw.name}.gates.json").write_text(json.dumps(gates, indent=2))
        if not gates["PASS"]:
            rec.update(status="gates_failed", failures=gates["failures"])
            return _fail(a, name, rec)
        target = ready / clips[0].name
        os.replace(clips[0], target)
        rec["clips"].append(dict(file=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                  bytes=target.stat().st_size, raw=str(raw), overlap=gates["overlap"],
                                  gates=gates,
                                  row=json.loads(report.read_text())["row"]))
    # Raw data and diagnostic logs remain until a verified-transfer receipt exists.
    rec.update(status="ok", end=time.time(), wall_s=round(time.time() - rec["start"], 1))
    complete.write_text(json.dumps(rec, indent=2))
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
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--p4d_repo", type=Path, default=Path("/root/point4d-datasets"))
    ap.add_argument("--src", type=Path, default=Path("/root/infinigen"))
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--split", default="pilot_v02")
    ap.add_argument("--overlap", choices=("all", "high", "medium", "low"), default="all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--collect-only", action="store_true", help="validate/package existing work scenes without regenerating")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--samples", type=int, default=128)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plan = [ln.split() for ln in a.plan.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda fs: run_one(fs[0], int(fs[1]), a), plan))
    if any(r is None or r.get("status") != "ok" for r in results):
        raise SystemExit("batch contains failed scenes; inspect batch.jsonl")
    print("BATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
