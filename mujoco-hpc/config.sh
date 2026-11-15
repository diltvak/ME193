# Edit these two lines, then everything else just works.
# Sourced by the scripts that run on your laptop.
export HPC_USER="dlitva01"    # <-- replace with your Tufts username (UTLN)
export HPC_HOST="login-prod.pax.tufts.edu"     # Tufts HPC login node; check research.tufts.edu if this has changed
export HPC_DIR="~/mujoco-hpc"             # where the files land on the cluster

# Guard: the scripts that source this file are for your LAPTOP. Slurm only exists on the cluster.
if command -v sbatch >/dev/null 2>&1; then
  echo "STOP: this script is for your laptop, but you are on the cluster ($(hostname))."
  echo "Switch to your laptop terminal (prompt does not say login-...), cd to mujoco-hpc, and run it there."
  exit 1
fi

# Stop early if the username has not been filled in.
if [[ "$HPC_USER" == YOUR_* ]]; then
  echo "Edit config.sh first: put your Tufts username in HPC_USER."
  exit 1
fi
