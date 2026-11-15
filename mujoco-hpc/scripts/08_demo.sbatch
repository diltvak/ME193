#!/bin/bash
#SBATCH --job-name=mujoco-demo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=mujoco-demo-%j.out

# Step 8: render a demo video on a GPU node.  Submit: sbatch 08_demo.sbatch
cd "$SLURM_SUBMIT_DIR"
PYMOD=$(cat "$HOME/mujoco-env/python_module" 2>/dev/null)
[ -n "$PYMOD" ] && module load "$PYMOD"
source "$HOME/mujoco-env/bin/activate"
export MUJOCO_GL=egl
python 08_demo.py
mkdir -p results && mv bounce.mp4 "results/bounce-$SLURM_JOB_ID.mp4"   # where 08b_fetch_results.sh looks
