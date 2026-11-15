# Run on the CLUSTER, in every new terminal:   source activate.sh
# (must be "source", not "bash", so it changes your current shell)
PYMOD=$(cat "$HOME/mujoco-env/python_module" 2>/dev/null)
[ -n "$PYMOD" ] && module load "$PYMOD"
source "$HOME/mujoco-env/bin/activate" && echo "ready: $(python --version) from $(which python)"
