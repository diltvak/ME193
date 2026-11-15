"""Two ping-pong paddles pass one ball back and forth. Rendered to a video (the cluster has no screen).

This is the same scene the reinforcement-learning task uses (see pingpong_duo_task/), with the paddles driven
by a hand-written rule instead of a learned policy, so you can see the physics first. The rule gets the
ball across about three times and then loses it; no amount of fiddling with its numbers made it reliable,
which is the reason to learn it instead:

    python3 runIt.py scripts/1112_pingpong_duo.py

    --still    keep both paddles still: the ball just dies on the paddle it was dropped on

It is 1105_pingpong.py with a second paddle. The new part: a level paddle can only send the ball straight up,
so each paddle gets two more joints that let it TILT toward its partner.

To TEACH the paddles with mjlab instead, see pingpong_duo_task/ (train.py, make_video.py).
"""
import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import imageio
import mujoco

SECONDS = 10.0             # how long to simulate
OUT = "pingpong_duo.mp4"   # the video file to write
STILL = "--still" in sys.argv

HOME_X = (-.5, .5)     # where the left and the right paddle sit
HOME_Z = .5            # height of both paddles when their joints are at 0
ARC_TOP = 1.5          # how high a pass should go (also the height the ball is dropped from)
G = 9.81


def paddle_xml(side, x, rgba):
    """One paddle: a thin disc (15 cm radius, 2 cm thick) that slides in x, y, z and tilts two ways."""
    return f"""
    <body name="{side}_paddle" pos="{x} 0 {HOME_Z}" gravcomp="1">
      <joint name="{side}_x" type="slide" axis="1 0 0" range="-.3 .3"/>
      <joint name="{side}_y" type="slide" axis="0 1 0" range="-.3 .3"/>
      <joint name="{side}_z" type="slide" axis="0 0 1" range="-.4 .4"/>
      <joint name="{side}_pitch" type="hinge" axis="0 1 0" range="-.6 .6" armature=".005"/>
      <joint name="{side}_roll" type="hinge" axis="1 0 0" range="-.6 .6" armature=".005"/>
      <geom type="cylinder" size=".15 .01" mass=".3" rgba="{rgba}" solref=".005 .5"/>
    </body>"""


def motors_xml(side):
    """Motors that pull each joint toward the position (or angle) we ask for (the "ctrl")."""
    slides = "".join(f"""
    <position name="{side}_{n}" joint="{side}_{n}" kp="3000" kv="60" forcerange="-150 150"/>""" for n in "xyz")
    tilts = "".join(f"""
    <position name="{side}_{n}" joint="{side}_{n}" kp="40" kv="1" forcerange="-5 5"/>""" for n in ("pitch", "roll"))
    return slides + tilts


# The world, described in MuJoCo's XML format. Keep these numbers in sync with pingpong_duo_task/pingpong_duo_task.py.
XML = f"""
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 -1 4"/>
    <camera name="side" pos="0 -3.8 .9" xyaxes="1 0 0 0 .1 1"/>

    <geom name="floor" type="plane" size="5 5 .1" rgba=".3 .4 .5 1"/>
    {paddle_xml("left", HOME_X[0], ".35 .6 1 1")}
    {paddle_xml("right", HOME_X[1], "1 .4 .4 1")}

    <body name="ball" pos="{HOME_X[0] + .03} -.02 {ARC_TOP}">
      <freejoint/>
      <geom type="sphere" size=".04" mass=".0027" rgba="1 .45 .1 1" solref=".005 .5"/>
    </body>
  </worldbody>

  <actuator>{motors_xml("left")}{motors_xml("right")}
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)   # build the physics model
data = mujoco.MjData(model)                   # the state: positions, velocities, time
renderer = mujoco.Renderer(model, 480, 640)   # draws pictures of the state

ball = model.body("ball")
qpos_ball = model.jnt_qposadr[ball.jntadr[0]]     # where the ball's x, y, z sit in data.qpos
dof_ball = model.jnt_dofadr[ball.jntadr[0]]       # ... and its velocities in data.qvel


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def controller():
    """Hand-written rule for both paddles. The learned policy replaces this."""
    if STILL:
        return
    bx, by, bz = data.qpos[qpos_ball:qpos_ball + 3]
    vx, vy, vz = data.qvel[dof_ball:dof_ball + 3]

    # Where, when and how fast will the ball come down onto a paddle at home height? (projectile motion, no air)
    drop = max(bz - HIT_Z, 0.0)
    vz_in = -math.sqrt(vz * vz + 2 * G * drop)     # its vertical speed when it gets there
    t = (vz - vz_in) / G                           # seconds until it gets there
    land_x, land_y = bx + vx * t, by + vy * t

    catcher = 0 if land_x < 0 else 1          # the paddle the ball is heading for; the other one waits
    for i, side in enumerate(("left", "right")):
        ctrl = {n: model.actuator(f"{side}_{n}").id for n in ("x", "y", "z", "pitch", "roll")}
        if i != catcher:                       # not my ball: go home, level out
            for n in ctrl:
                data.ctrl[ctrl[n]] = 0.0
            continue

        # 1. Get under the spot where the ball will land.
        px = clamp(land_x - HOME_X[i], -.25, .25)
        py = clamp(land_y, -.25, .25)
        data.ctrl[ctrl["x"]], data.ctrl[ctrl["y"]] = px, py

        # 2. Tilt like a mirror. The ball should leave with the velocity that carries it over the top of
        #    the arc and down onto the other paddle. A hit only pushes the ball along the paddle's normal
        #    (the direction its face points), so the face must point along (outgoing - incoming) velocity.
        vz_out = math.sqrt(2 * G * (ARC_TOP - HIT_Z))
        flight = 2 * vz_out / G
        out_x = (HOME_X[1 - i] - (HOME_X[i] + px)) / flight
        out_y = (0.0 - py) / flight
        nx, ny, nz = out_x - vx, out_y - vy, vz_out - vz_in
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / length, ny / length, nz / length
        data.ctrl[ctrl["pitch"]] = clamp(math.atan2(nx, nz), -.5, .5)
        data.ctrl[ctrl["roll"]] = clamp(-math.asin(ny), -.5, .5)

        # 3. Swing up into the ball. The contact is "dead": the ball leaves a still paddle with only BOUNCE
        #    times the speed it came in with, so the paddle must be rising at the right speed when they meet:
        #        speed out = BOUNCE * speed in + (1 + BOUNCE) * paddle speed     (all measured along the normal)
        speed_in = -(vx * nx + vy * ny + vz_in * nz)
        speed_out = out_x * nx + out_y * ny + vz_out * nz
        rise = (speed_out - BOUNCE * speed_in) / ((1 + BOUNCE) * nz)
        #    Wait a little below home, then rise at that speed so as to be at home exactly when the ball arrives.
        data.ctrl[ctrl["z"]] = -rise * min(t, RUN_UP)


HIT_Z = HOME_Z + .05   # height of the ball's centre when it sits on a paddle at home
BOUNCE = .1            # found by trial and error, and touchy, which is why we learn the swing instead
RUN_UP = .06           # s the paddle spends rising before the hit

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
