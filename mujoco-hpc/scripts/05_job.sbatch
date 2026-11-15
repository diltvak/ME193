#!/bin/bash
#SBATCH --job-name=mujoco-test
#SBATCH --partition=batch        # change if 02_check_cluster.sh shows a different default
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=mujoco-test-%j.out

# Step 5: run the smoke test on a compute node.  Submit with:  sbatch 05_job.sbatch
# Watch with: squeue -u $USER      Read result: cat mujoco-test-<id>.out
cd "$SLURM_SUBMIT_DIR"
# load the same python module the venv was built with (recorded by 03_setup_env.sh)
PYMOD=$(cat "$HOME/mujoco-env/python_module" 2>/dev/null)
[ -n "$PYMOD" ] && module load "$PYMOD"
source "$HOME/mujoco-env/bin/activate"
python 04_test_mujoco.py
