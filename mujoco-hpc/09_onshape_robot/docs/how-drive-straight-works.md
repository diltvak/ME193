# How Drive-Straight works

A reference for when you come back to this in a few months. The code is one file,
[`../mjlab_task/drive_task.py`](../mjlab_task/drive_task.py); everything below describes what is in it.

## In five sentences

Drive-Straight is a reinforcement-learning task: a policy has to learn to drive your two-wheeled robot in a straight line
down the x axis. Each practice run starts the robot at the origin, pointing up to 0.4 radians off-axis, and lasts 10
seconds. The policy sees the robot's own velocities, heading error, sideways drift and last command, and outputs one speed
for each wheel. Rewards pay for moving forward, staying on the line y = 0, facing along +x and keeping commands smooth, and
a run ends early if the robot tips over or wanders 1.5 m off the line. Training runs thousands of these robots at once on
the GPU and gradually adjusts the policy toward whatever scored the best reward.

## One step of the loop (50 times a second)

| | What | In the code |
|---|---|---|
| **See** (13 numbers) | forward/sideways/up speed (3), turning speed (3), heading and sideways offset from the line (3), wheel speeds (2), the previous command (2) | `actor_terms` in `drive_straight_env_cfg` |
| **Think** | a small neural network (two layers of 64) maps the 13 numbers to 2 numbers | `drive_straight_ppo_runner_cfg` |
| **Act** | the 2 outputs, between -1 and 1, times `MAX_WHEEL_SPEED` (10 rad/s), are the wheel speeds to aim for | `JointVelocityActionCfg` |

## What earns points (per step, summed over the 10-second run)

| Term | Weight | Meaning |
|---|---|---|
| `forward_speed` | +1.0 | 0 when standing still, 1 at `TARGET_SPEED`, less beyond it, negative backwards |
| `stay_on_line` | +1.0 | minus the square of the sideways distance from y = 0 |
| `face_forward` | +0.5 | minus the square of the heading error |
| `action_rate` | -0.05 | minus the square of how much the command changed since the last step |

A run ends (and restarts) when: the 10 seconds are up, the robot tilts more than 60 degrees, or it is more than 1.5 m from the line.

## What happens when you train

1. `runIt.py` copies `train.py`, `drive_task.py` and the `onshape/` robot to a fresh folder on the cluster and submits a GPU job.
2. `train.py` imports `drive_task.py`, which registers the task name `Drive-Straight` with mjlab, then starts mjlab's trainer.
3. The trainer builds 4096 copies of the robot (`--env.scene.num-envs`) and repeats an **iteration**: every robot drives
   24 steps with the current policy, then the policy (PPO, an algorithm for this) is nudged toward the actions that
   scored well. `max_iterations` is 500 unless you pass `--agent.max-iterations`.
4. Every 50 iterations it saves a **checkpoint** (`model_<n>.pt`: the policy's numbers). `runIt.py` copies the whole folder back
   to `runs/train-<time>/`.
5. `make_video.py` loads a checkpoint, drives one robot for 10 seconds and records a movie plus a table of x, y and heading.

The copy of `drive_task.py` inside each `runs/train-*/` folder is a snapshot of what that run used. Edit only the one in `mjlab_task/`.

## Words

| Word | Meaning here |
|---|---|
| policy | the neural network that turns what the robot sees into wheel speeds. It is what gets trained. |
| iteration | one round of "everyone drives a bit, then the policy improves a little" |
| checkpoint | a saved copy of the policy (`model_499.pt`), usable later without retraining |
| PPO | the learning algorithm (proximal policy optimization) that updates the policy |
| mjlab | the library that runs thousands of MuJoCo robots on a GPU and does the training |

## Results so far

| Robot and training | Forward in 10 s | Sideways (y) | Heading |
|---|---|---|---|
| Sample robot (8 cm wheels), 300 iterations | 4.7 m (0.5 m/s) | within 0.1 m | within 2 degrees |
| Author's rod-wheel robot, 500 iterations | 1.15 m | down to -0.5 m | wanders -43 to +38 degrees |
| Author's rod-wheel robot, 2000 iterations | 1.05 m | within about 0.25 m | within about 27 degrees |

The sample robot drives straight at the target speed after about a minute of training. The author's own robot, with jerky
rod wheels, crawls at about 0.1 m/s and still wobbles. The `TARGET_SPEED` of 0.5 m/s suits the sample robot but is more than
the rod-wheel robot can do (driving both wheels at 8 rad/s by hand moves it only about 0.03 m/s), so for that robot the
speed reward just keeps paying for more speed.

## Ideas for next time

- Lower `TARGET_SPEED` to about 0.15 so it learns a steady speed.
- A gentler `WHEEL_TORQUE_LIMIT`, or heavier `face_forward` and `stay_on_line` weights, to calm the wobble.
- Randomize wheel friction and mass during training so the policy copes with the jerky rod wheels.
- Train longer (`--agent.max-iterations 2000` takes about 13 minutes).
