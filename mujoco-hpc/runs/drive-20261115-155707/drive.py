"""Drive a two-wheeled robot that was built in Onshape, and film it. Run it with runIt.py (see README.md).

Everything you may want to change is in the first block of constants.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import imageio
import mujoco

MODEL = sys.argv[1] if len(sys.argv) > 1 else "onshape/scene.xml"   # your Onshape robot (or pass sample/scene.xml)
LEFT, RIGHT = "left_wheel", "right_wheel"   # the wheel actuator names (printed below if they are wrong)
OUT = "drive.mp4"

# What the wheels do, in order: (until this second, left wheel speed, right wheel speed), in radians/second.
PLAN = [
    (1.0, 0, 0),      # sit still and settle
    (4.0, 8, 8),      # both wheels forward: drive straight
    (7.0, 4, -4),     # wheels opposite: spin on the spot
    (10.0, 8, 4),     # left wheel faster: curve to the right
]

# ---- load the robot -------------------------------------------------------------------------------
spec = mujoco.MjSpec.from_file(MODEL)
chassis = spec.worldbody.first_body()
if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
    chassis.add_freejoint()           # let the robot move around (an Onshape export may pin it in place)
model = spec.compile()
data = mujoco.MjData(model)

print("joints:   ", [model.joint(i).name for i in range(model.njnt) if model.joint(i).name])
print("actuators:", [model.actuator(i).name for i in range(model.nu)])
left, right = model.actuator(LEFT).id, model.actuator(RIGHT).id   # stops with a clear error if a name is wrong

# Put the robot just above the floor: find its lowest point and lift it from there.
mujoco.mj_forward(model, data)
lowest = (data.geom_xpos[:, 2] - model.geom_rbound).min()
data.qpos[2] += 0.01 - lowest

# Only the wheels should grip; the chassis and skid should slide.
wheel_bodies = set(model.jnt_bodyid[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE])
for g in range(model.ngeom):
    if model.geom_bodyid[g] != 0 and model.geom_bodyid[g] not in wheel_bodies:
        model.geom_friction[g, 0] = 0.01


def drive(actuator, speed):
    """Tell one wheel to turn at `speed` rad/s for one physics step."""
    if model.actuator_biasprm[actuator, 1] != 0:           # a position motor: keep moving its target angle
        data.ctrl[actuator] += speed * model.opt.timestep
    else:                                                    # a velocity motor: give it the speed
        data.ctrl[actuator] = speed


# ---- simulate and film ----------------------------------------------------------------------------
renderer = mujoco.Renderer(model, 480, 640)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING     # follow the robot
camera.trackbodyid = model.body(chassis.name).id
camera.distance, camera.azimuth, camera.elevation = 0.9, 140, -25

FPS = 30
frames = []
for end, left_speed, right_speed in PLAN:
    while data.time < end:
        while data.time < len(frames) / FPS:
            drive(left, left_speed)
            drive(right, right_speed)
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())
    x, y = data.qpos[0], data.qpos[1]
    print(f"t={data.time:4.1f}s  robot at x={x:+.2f} m, y={y:+.2f} m")

renderer.close()
imageio.mimwrite(OUT, frames, fps=FPS)
print(f"wrote {OUT}: {len(frames)} frames")
