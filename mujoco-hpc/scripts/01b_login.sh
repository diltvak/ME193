#!/usr/bin/env bash
# Step 2.5 (run on your LAPTOP, in a SECOND terminal): log in to the cluster
# and land in ~/mujoco-hpc. Everything after this is typed at the cluster prompt.
# Reuses the connection from step 1, so it usually skips the Duo prompt.
source "$(dirname "$0")/../config.sh"
mkdir -p ~/.ssh/sockets
echo "Logging in to $HPC_HOST. When the prompt changes to $HPC_USER@<something>, you are on the cluster."
echo "Type 'exit' to come back to your laptop."
exec ssh -t -o ControlMaster=auto -o ControlPath="$HOME/.ssh/sockets/%r@%h" -o ControlPersist=10m \
    "$HPC_USER@$HPC_HOST" "cd $HPC_DIR 2>/dev/null || echo 'note: $HPC_DIR not there yet, run step 2 first'; exec \$SHELL -l"
