"""A ball drops onto a ping-pong paddle. Rendered to a video (the cluster has no screen).

This is the same scene the reinforcement-learning task uses (see pingpong_task/), with the paddle driven
by a simple hand-written rule instead of a learned policy, so you can see the physics first:

    python3 runIt.py scripts/1105_pingpong.py

    --still    keep the paddle still: the ball just dies on it (the contact soaks up the energy)

To TEACH the paddle with mjlab instead, see pingpong_task/ (train.py, make_video.py).
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import imageio
import mujoco

SECONDS = 8.0          # how long to simulate
OUT = "pingpong.mp4"   # the video file to write
STILL = "--still" in sys.argv

# The world, described in MuJoCo's XML format. Keep these numbers in sync with pingpong_task/pingpong_task.py.
XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 -1 4"/>
    <camera name="side" pos="0 -3.2 .9" xyaxes="1 0 0 0 .1 1"/>

    <geom name="floor" type="plane" size="5 5 .1" rgba=".3 .4 .5 1"/>

    <!-- The paddle: a thin disc (15 cm radius, 2 cm thick) that slides in x, y and z. -->
    <body name="paddle" pos="0 0 .5" gravcomp="1">
      <joint name="paddle_x" type="slide" axis="1 0 0" range="-.3 .3"/>
      <joint name="paddle_y" type="slide" axis="0 1 0" range="-.3 .3"/>
      <joint name="paddle_z" type="slide" axis="0 0 1" range="-.4 .4"/>
      <geom type="cylinder" size=".15 .01" mass=".3" rgba=".9 .9 .95 1" solref=".005 .5"/>
    </body>

    <body name="ball" pos=".03 -.02 1.5">
      <freejoint/>
      <geom type="sphere" size=".04" mass=".0027" rgba="1 .45 .1 1" solref=".005 .5"/>
    </body>
  </worldbody>

  <!-- A motor that pulls each slide joint toward the position we ask for (the "ctrl"). -->
  <actuator>
    <position name="x" joint="paddle_x" kp="3000" kv="60" forcerange="-150 150"/>
    <position name="y" joint="paddle_y" kp="3000" kv="60" forcerange="-150 150"/>
    <position name="z" joint="paddle_z" kp="3000" kv="60" forcerange="-150 150"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)   # build the physics model
data = mujoco.MjData(model)                   # the state: positions, velocities, time
renderer = mujoco.Renderer(model, 480, 640)   # draws pictures of the state

ball = model.body("ball")
qpos_ball = model.jnt_qposadr[ball.jntadr[0]]     # where the ball's x, y, z sit in data.qpos
dof_ball = model.jnt_dofadr[ball.jntadr[0]]       # ... and its velocities in data.qvel


def controller():
    """Hand-written paddle rule. The learned policy replaces this."""
    if STILL:
        return
    bx, by, bz = data.qpos[qpos_ball:qpos_ball + 3]
    vz = data.qvel[dof_ball + 2]
    data.ctrl[0] = max(-.25, min(.25, bx))      # stay under the ball
    data.ctrl[1] = max(-.25, min(.25, by))
    # Ball falling and close above the paddle: swing up into it. Otherwise drop back down and wait.
    # (Found by trial and error: .02 gives bounces of 1.0-1.4 m; .04 sends the ball 3-6 m up. Touchy, so we learn it.)
    data.ctrl[2] = .02 if (bz - (.5 + data.qpos[2]) < .35 and vz < 0) else -.2


FPS = 30
frames = []
while data.time < SECONDS:
    # Advance the physics by one video frame (many small steps), then take a picture.
    while data.time < len(frames) / FPS:
        controller()
        mujoco.mj_step(model, data)
    renderer.update_scene(data, camera="side")
    frames.append(renderer.render())

renderer.close()
imageio.mimwrite(OUT, frames, fps=FPS)
print(f"wrote {OUT}: {len(frames)} frames, {SECONDS:g} s of simulation")
