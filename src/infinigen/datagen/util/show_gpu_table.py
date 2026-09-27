# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import functools
import logging
import os
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from itertools import chain
from shutil import which

gres_regex = re.compile(".*gpu:([^:]+):([0-9]+).*").fullmatch
cpu_regex = re.compile(".+/([0-9]+)[^/]+").fullmatch

logger = logging.getLogger(__name__)

SINFO_MAX_RETRIES = 10


@functools.cache
def slurm_available() -> bool:
    """True only if `sinfo` exists AND can reach a SLURM controller.

    Container images (e.g. runpod/pytorch) ship SLURM client binaries without a cluster;
    `sinfo` then fails with "Could not establish a configuration source". Previously
    sinfo() retried that forever (60 s sleeps), so manage_jobs stalled after the first task.
    Set INFINIGEN_DISABLE_SLURM=1 to skip the probe entirely.
    """
    if os.environ.get("INFINIGEN_DISABLE_SLURM", "0") not in ("", "0", "false", "False"):
        return False
    exe = which("sinfo")
    if exe is None:
        return False
    try:
        subprocess.run(
            [exe, "--noheader", "--format=%N"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning(
            f"{exe} is installed but SLURM is unreachable ({e!r}); treating as no SLURM"
        )
        return False


def sinfo():
    if not slurm_available():
        return ""
    sinfo_command = [which("sinfo"), "--Node", "--format=%12N%22P%C%30G%10m", "--noheader"]
    for attempt in range(SINFO_MAX_RETRIES):
        try:
            return subprocess.check_output(sinfo_command, timeout=60).decode()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            current_time_str = datetime.now().strftime("%m/%d %I:%M%p")
            print(f"[{current_time_str}] sinfo failed ({attempt + 1}/{SINFO_MAX_RETRIES}) with error:\n{e}")
            time.sleep(60)
    raise RuntimeError(f"sinfo failed {SINFO_MAX_RETRIES} times in a row")


def get_gpu_nodes():
    sinfo_output = sinfo()
    gpu_table = {}
    node_type_lookup = defaultdict(set)
    shared_node_mem = {}
    for line in sinfo_output.splitlines():
        node, group, gres, totalmem = line.split()
        if group != "all":
            if group in {"pvl", "cs*", "pnlp"}:
                num_cpus = int(cpu_regex(gres).group(1))
                shared_node_mem[node] = int((int(totalmem) / 1024) / num_cpus)
            if gres_regex(gres):
                gpu, num = gres_regex(gres).groups()
                if gpu not in gpu_table:
                    gpu_table[gpu] = defaultdict(int)
                gpu_table[gpu][group] += int(num)
                node_type_lookup[gpu].add(node)

    return gpu_table, dict(node_type_lookup), shared_node_mem


# e.g. nodes_with_gpus('gtx_1080', 'k80')
def nodes_with_gpus(*gpu_names):
    if not slurm_available():
        return []
    if len(gpu_names) == 0:
        return []
    _, node_type_lookup, _ = get_gpu_nodes()
    return sorted(
        chain.from_iterable(node_type_lookup.get(n, set()) for n in gpu_names)
    )


if __name__ == "__main__":
    gpu_table, node_type_lookup, shared_node_mem = get_gpu_nodes()
    for group, lookup in gpu_table.items():
        print(f"{group.ljust(10)} {dict(lookup)} Total: {sum(lookup.values())}")
    print()

    for k, v in sorted(node_type_lookup.items()):
        print(f"{k.ljust(10)} {','.join(v)}")
    print()

    for key, val in shared_node_mem.items():
        print(f"{key} {val}gb per cpu")
