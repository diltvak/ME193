#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
PYMOD=$(cat "$HOME/mujoco-env/python_module" 2>/dev/null)
[ -n "$PYMOD" ] && module load "$PYMOD"
source "$HOME/mujoco-env/bin/activate"
export MUJOCO_GL=egl
python train.py Ping-Pong-Duo --env.scene.num-envs 4096
