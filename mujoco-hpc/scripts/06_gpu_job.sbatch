#!/bin/bash
#SBATCH --job-name=mujoco-gpu
#SBATCH --partition=gpu          # check names with: sinfo -s   (Tufts GPU queue is usually "gpu")
#SBATCH --gres=gpu:1             # ask for one GPU; add the type (gpu:a100:1) if you need a specific one
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=mujoco-gpu-%j.out

# Step 7: same smoke test on a GPU node, with EGL (GPU) rendering.  Submit: sbatch 06_gpu_job.sbatch
cd "$SLURM_SUBMIT_DIR"
PYMOD=$(cat "$HOME/mujoco-env/python_module" 2>/dev/null)
[ -n "$PYMOD" ] && module load "$PYMOD"
source "$HOME/mujoco-env/bin/activate"

echo "== GPU =="; nvidia-smi -L || echo "no nvidia-smi: not a GPU node?"
export MUJOCO_GL=egl
python 04_test_mujoco.py
