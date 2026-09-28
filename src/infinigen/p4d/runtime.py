"""Keep Blender worker pools within the CPU allocation advertised by a container."""
import os
import hashlib
import json
import subprocess
from pathlib import Path


def source_identity():
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.read_bytes())
    identity = dict(generator_code_sha256=digest.hexdigest(), fork_commit=None)
    manifest = package / "source_manifest.json"
    if manifest.exists():
        stamped = json.loads(manifest.read_text())
        if stamped["generator_code_sha256"] == identity["generator_code_sha256"]:
            return stamped
    root = package.parents[2]
    if (root / ".git").exists():
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True)
        identity.update(fork_commit=revision.stdout.strip() or None, dirty=bool(dirty.stdout.strip()))
    else:
        identity["declared_fork_commit"] = os.environ.get("INFINIGEN_FORK_REVISION")
    return identity


def configure_cpu_budget():
    if not hasattr(os, "sched_getaffinity"):
        return
    allowed = sorted(os.sched_getaffinity(0))
    quota = None
    try:
        maximum = Path("/sys/fs/cgroup/cpu.max")
        if maximum.exists():
            value, period = maximum.read_text().split()
            if value != "max":
                quota = float(value) / float(period)
        else:
            root = Path("/sys/fs/cgroup/cpu")
            value = int((root / "cpu.cfs_quota_us").read_text())
            if value > 0:
                quota = value / int((root / "cpu.cfs_period_us").read_text())
    except (OSError, ValueError):
        pass
    count = int(os.environ.get("P4D_CPU_THREADS", max(1, int(quota)) if quota else len(allowed)))
    count = max(1, min(count, len(allowed)))
    os.sched_setaffinity(0, allowed[:count])
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", str(count))
