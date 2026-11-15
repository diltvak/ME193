#!/bin/bash
#SBATCH --job-name=drive-straight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=train-%j.out

# Train Drive-Straight on a GPU node.  Submit from this folder on the cluster:  sbatch train.sbatch
cd "$SLURM_SUBMIT_DIR"
PYMOD=$(cat "$HOME/mujoco-env/python_module" 2>/dev/null)
[ -n "$PYMOD" ] && module load "$PYMOD"
source "$HOME/mujoco-env/bin/activate"
nvidia-smi -L
python train.py Drive-Straight --env.scene.num-envs 4096
