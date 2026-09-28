#!/bin/bash
# Install the infinigen-fixes fork (branch p4d-fixes) on a Runpod ubuntu24.04 pod
# (image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404): legacy 1.x pipeline + terrain + OpenGL customgt + v2.
# Expects the source at $SRC (default /root/infinigen; rsynced, or `git clone --recurse-submodules <fork> -b p4d-fixes`).
# All former install-time patches are now commits on the branch; the only runtime guard left is SLURM:
# the fork's datagen already tolerates unreachable SLURM, and INFINIGEN_DISABLE_SLURM=1 skips the probe entirely.
set -euo pipefail
SRC=${SRC:-/root/infinigen}
exec > /root/install.log 2>&1
T0=$(date +%s)
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q wget cmake g++ git libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev zlib1g-dev \
   libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6 libgl1 libegl1 ffmpeg \
   libxinerama-dev libxcursor-dev libxi-dev libxrandr-dev time rsync zstd
echo "APT_DONE $(( $(date +%s)-T0 ))"
cd "$SRC"
uv venv --python 3.11 --seed /root/venv
. /root/venv/bin/activate
INFINIGEN_MINIMAL_INSTALL=False INFINIGEN_INSTALL_TERRAIN=True INFINIGEN_INSTALL_CUSTOMGT=True \
  pip install -e ".[infinigen1,terrain,vis,sim]" -v > /root/pip.log 2>&1
echo "PIP_EXIT $? $(( $(date +%s)-T0 ))"
pip install -q OpenEXR zstandard boto3 pytest pybullet==3.2.7 >> /root/pip.log 2>&1
ls -la src/infinigen/datagen/customgt/build/customgt
python -c "import bpy, infinigen, infinigen2; print('BPY', bpy.app.version_string)"
echo "INSTALL_DONE $(( $(date +%s)-T0 ))"
