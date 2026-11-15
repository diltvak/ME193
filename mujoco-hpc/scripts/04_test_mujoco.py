"""Headless MuJoCo smoke test. No display needed.

Run it on the login node (quick) and again through Slurm (05_job.sbatch).
Each check prints PASS/FAIL; rendering is optional and only warns.
"""
import contextlib
import os
import platform
import socket
import sys
import time

# Offscreen rendering backend must be chosen BEFORE mujoco is imported.
# Override from the shell:  MUJOCO_GL=osmesa python 04_test_mujoco.py
os.environ.setdefault("MUJOCO_GL", "egl")

print(f"host {socket.gethostname()}  python {platform.python_version()}  "
      f"slurm job {os.environ.get('SLURM_JOB_ID', '(none: login node)')}")

try:
    import mujoco
    import numpy as np
except ImportError as e:
    sys.exit(f"FAIL import: {e}\n  running {sys.executable}\n"
             "  New terminal? Run:  source ~/mujoco-env/bin/activate   (then try again)\n"
             "  Never ran step 4?   bash 03_setup_env.sh")
print("PASS import mujoco", mujoco.__version__)

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# A box dropped from 1 m should come to rest on the plane (z ~ 0.1).
t0 = time.time()
steps = 5000
for _ in range(steps):
    mujoco.mj_step(model, data)
dt = time.time() - t0
z = data.qpos[2]
ok = abs(z - 0.1) < 0.01
print(f"{'PASS' if ok else 'FAIL'} physics: box at rest z={z:.3f} (expect 0.100)")
print(f"     {steps} steps in {dt:.2f}s = {steps / dt:,.0f} steps/s")

@contextlib.contextmanager
def quiet_stderr():
    """Hide the C-level warnings EGL prints on a machine without a GPU ("No 3D enabled", "failed to create dri2
    screen"). They are harmless, and the message printed below says what happened."""
    sys.stderr.flush()
    saved = os.dup(2)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)


# Optional: offscreen rendering. egl needs a GPU node; osmesa is the CPU fallback.
gl = os.environ["MUJOCO_GL"]
with quiet_stderr():
    try:
        r = mujoco.Renderer(model, 120, 160)
        mujoco.mj_forward(model, data)
        r.update_scene(data)
        print(f"PASS render via {gl}: image {r.render().shape}")
        r.close()  # free the GL context now; leaving it to interpreter exit prints harmless EGL tracebacks
    except Exception as e:
        print(f"note: render via {gl} unavailable ({type(e).__name__}); fine for physics/training. "
              "Login nodes have no GPU; try again inside 05_job.sbatch or with MUJOCO_GL=osmesa")
print("done")
