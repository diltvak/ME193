#!/usr/bin/env bash
# Step 3 (runs ON the cluster login node): survey what is available.
# Read-only; installs nothing.

echo "== Who/where =="
echo "user: $(whoami)   host: $(hostname)   date: $(date)"

echo; echo "== Slurm (the job scheduler) =="
if command -v sbatch >/dev/null; then sbatch --version; else echo "MISSING: sbatch not found"; fi

echo; echo "== Partitions (queues). Look for 'gpu' if you want GPUs =="
sinfo -s 2>/dev/null || echo "sinfo failed"

echo; echo "== Your account / allowed partitions =="
sacctmgr -nP show assoc user="$(whoami)" format=account,partition,qos 2>/dev/null | head || true

echo; echo "== Python on the login node =="
for p in python3 python; do command -v $p >/dev/null && { echo "$p -> $(command -v $p): $($p --version 2>&1)"; }; done

echo; echo "== Modules (python / anaconda / cuda / mesa) =="
if command -v module >/dev/null 2>&1 || type module >/dev/null 2>&1; then
  module avail 2>&1 | grep -iE 'python|anaconda|miniconda|cuda|mesa|osmesa|ffmpeg' | head -30
else
  echo "no 'module' command (try: source /etc/profile.d/modules.sh)"
fi

echo; echo "== Is MuJoCo (or related) already installed on the cluster? =="
found=$(module -t avail 2>&1 | grep -iE 'mujoco|mjx|gymnasium|brax|isaac|robosuite' | head -10)
if [ -n "$found" ]; then echo "preinstalled modules:"; echo "$found"
else echo "no MuJoCo module on this cluster (expected). 03_setup_env.sh installs it in your own venv."; fi

echo; echo "== Disk =="
df -h "$HOME" | tail -1
quota -s 2>/dev/null | head -5 || true

echo; echo "== Internet from login node (needed for pip install) =="
if curl -sI --max-time 8 https://pypi.org >/dev/null; then echo "ok: pypi reachable"; else echo "FAIL: cannot reach pypi"; fi

echo; echo "Next, on the cluster:  bash 03_setup_env.sh"
