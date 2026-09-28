"""Keep Blender worker pools within the CPU allocation advertised by a container."""
import os
from pathlib import Path


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
