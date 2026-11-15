#!/usr/bin/env bash
# Step 4 (runs ON the cluster login node): make a Python venv with MuJoCo and mjlab (reinforcement learning).
# mjlab needs Python 3.11+; the cluster's default is 3.9, too old. It also brings PyTorch (about 6 GB),
# and it chooses the MuJoCo version itself, so MuJoCo is installed as part of mjlab, not separately.
# By default this loads the newest python module it can find. To choose one
# yourself, put its name in PY_MODULE (names come from 02_check_cluster.sh).
PY_MODULE=""
VENV="$HOME/mujoco-env"

set -e
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || true

if [ -z "$PY_MODULE" ]; then
  # 'module avail' prints to stderr; keep names like python/3.12.1, take the highest >= 3.11
  PY_MODULE=$(module -t avail python 2>&1 | grep -E '^python/3\.[0-9]+' \
    | grep -viE 'rc|alpha|beta' \
    | awk '{ if (match($0, /3\.[0-9]+(\.[0-9]+)?/)) { v=substr($0,RSTART,RLENGTH); split(v,a,"."); if (a[2]>=11) print a[2]*1000+a[3], $0 } }' \
    | sort -n | tail -1 | cut -d' ' -f2 | tr -d ':')
fi
if [ -n "$PY_MODULE" ]; then echo "loading module: $PY_MODULE"; module load "$PY_MODULE"
else echo "No python 3.11+ module found; using system python3 (may be too old)."; fi

python3 --version
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' || {
  echo "FAIL: python is older than 3.11. Run 'module avail python', set PY_MODULE in this file, rerun."; exit 1; }

echo "== Disk quota (the install needs about 8 GB free) =="
quota -s 2>/dev/null | head -5 || true
echo "If 'space' is within 8 GB of 'quota', clear files first (du -sh ~/* | sort -h), then rerun."

rm -rf "$VENV"            # start clean in case an earlier attempt left a broken venv
python3 -m venv "$VENV"
echo "$PY_MODULE" > "$VENV/python_module"   # so 05_job.sbatch loads the same python
source "$VENV/bin/activate"
python -m pip install --no-cache-dir --upgrade pip
# --no-cache-dir: pip would otherwise keep a second copy of every download in ~/.cache (several GB of your quota).
# The cluster's GPU driver is CUDA 12.9, but the newest PyTorch and Warp on PyPI are built for CUDA 13 and fail with
# "NVIDIA driver too old". So install the CUDA 12 builds FIRST (torch 2.11 from PyTorch's cu128 index; warp 1.14);
# mjlab accepts them (torch>=2.7, warp>=1.14) and so does not fetch the CUDA 13 ones.
python -m pip install --no-cache-dir "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
python -m pip install --no-cache-dir "warp-lang==1.14.0"
# --only-binary mujoco: never compile MuJoCo from source (needs MUJOCO_PATH and a C++ toolchain).
python -m pip install --no-cache-dir --only-binary mujoco mjlab numpy imageio imageio-ffmpeg
python -m pip check
python -c "import torch; assert torch.version.cuda.startswith('12'), 'wrong PyTorch build: ' + torch.version.cuda"
python -c "import mujoco, mjlab, torch; print('mujoco', mujoco.__version__, '+ mjlab, torch', torch.__version__, 'installed in', '$VENV')"
echo; echo "Next: python 04_test_mujoco.py   (quick, on the login node)"
echo "Then: sbatch 05_job.sbatch        (real test, on a compute node)"
