"""A ball drops onto the floor and bounces. Rendered to a video (the cluster has no screen).

    python 08_demo.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import imageio
import mujoco

SECONDS = 4.0          # how long to simulate
OUT = "bounce.mp4"     # the video file to write

# The world, described in MuJoCo's XML format.
XML = """
<mujoco>
  <worldbody>
    <light pos="0 -1 4"/>
    <camera name="side" pos="0 -3 1.5" xyaxes="1 0 0 0 .3 1"/>

    <geom name="floor" type="plane" size="5 5 .1" rgba=".3 .4 .5 1"/>

    <body name="ball" pos="0 0 1.5">
      <freejoint/>
      <!-- solref = "-stiffness -damping" of the contact. Less damping = bouncier. -->
      <geom type="sphere" size=".1" rgba="1 .3 .2 1" solref="-20000 -10"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)   # build the physics model
data = mujoco.MjData(model)                   # the state: positions, velocities, time
renderer = mujoco.Renderer(model, 480, 640)   # draws pictures of the state

FPS = 30
frames = []
while data.time < SECONDS:
    # Advance the physics by one video frame (many small steps), then take a picture.
    while data.time < len(frames) / FPS:
        mujoco.mj_step(model, data)
    renderer.update_scene(data, camera="side")
    frames.append(renderer.render())

renderer.close()
imageio.mimwrite(OUT, frames, fps=FPS)
print(f"wrote {OUT}: {len(frames)} frames, {SECONDS:g} s of simulation")
