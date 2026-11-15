# Train the robot to drive straight with mjlab

`drive_task.py` turns your Onshape robot (`../onshape/robot.xml`) into an [mjlab](https://github.com/mujocolab/mjlab)
reinforcement-learning task. 4096 copies of the robot start slightly off-angle and learn to drive down +x while staying
on the line y = 0. The policy outputs one speed per wheel.

mjlab trains only on an NVIDIA GPU, so training runs on the cluster. (On a Mac it can only play back.)

| File | What it is |
|---|---|
| `drive_task.py` | The task: robot, rewards, resets. **Tune this one.** |
| `train.py`, `play.py` | mjlab's own train/play commands, with our task registered first |
| `make_video.py` | Records a movie of a trained policy on the cluster (step 4) |
| `train.sbatch` | A Slurm job that trains on a GPU, for running by hand on the cluster |

How it works in plain words, with the rewards and the words used (policy, iteration, checkpoint): [../docs/how-drive-straight-works.md](../docs/how-drive-straight-works.md).

Everything runs from your laptop with `runIt.py`, in the `mujoco-hpc` folder (venv active, VPN on if off campus).

## 1. Install

Nothing extra: `python3 setup.py` (the same setup as the bouncing-ball demo) already installed mjlab on the cluster.
If you set up before mjlab was added, rerun `python3 setup.py --reinstall`.

## 2. Check the task loads (short run)

```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
  09_onshape_robot/mjlab_task/train.py Drive-Straight --env.scene.num-envs 64
```
Under "job output" you should see training iterations with a mean reward. If it stops with an error instead, the
message says what to fix in `drive_task.py` (see Tuning).

## 3. Train

Same command with more robots and time (`runIt.py` waits until the job ends, so keep the terminal open). A few minutes
is enough to see it learn; add `--agent.max-iterations 2000` (about 13 minutes, use `--time 00:45:00`) for a better driver:
```
python3 runIt.py --gpu --time 00:15:00 --mem 16G --cpus 4 \
  --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
  09_onshape_robot/mjlab_task/train.py Drive-Straight --env.scene.num-envs 4096
```
The logs and checkpoints (`model_<n>.pt`, saved every 50 iterations) come back to
`runs/train-<time>/logs/rsl_rl/drive_straight/<date>/` on your laptop. The last one is `model_499.pt` for the default 500
iterations, `model_1999.pt` for 2000.

## 4. Watch the result: a movie made on the cluster

`make_video.py` loads a checkpoint, drives one robot for 10 seconds on a GPU node, records an `.mp4`, and prints where the
robot is each second (x forward, y sideways, heading; straight driving keeps y and heading near 0). `runIt.py` brings the
movie home and opens it. Send it your checkpoint with a third `--with` (find the file with `ls runs/train-*/logs/rsl_rl/drive_straight/*/`):
```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
  --with runs/train-XXXXXXXX-XXXXXX/logs/rsl_rl/drive_straight/XXXX-XX-XX_XX-XX-XX/model_499.pt \
  09_onshape_robot/mjlab_task/make_video.py model_499.pt
```
Do not type the `X`s. They stand for the folder names in your own `runs/` folder: type `runs/train-` and press Tab to
complete each name. Use the checkpoint name you have (`model_1999.pt` if you trained 2000 iterations) in both places.

Optional, watching live on your Mac instead: mjlab's `play.py` needs **Python 3.11 to 3.13** (with 3.14 pip only finds a
very old mjlab and it fails with `No module named 'mjlab.actuator'`). Make an environment with Python 3.13
(`python3.13 -m venv ~/mjlab-play`, activate it, `python -m pip install mjlab`) and run `python play.py Drive-Straight
--checkpoint-file <your model file>` from this folder. You do not need this for the movie.

## Tuning (all in `drive_task.py`)

- **Wheels barely turn or the robot crawls:** raise `WHEEL_KD` or `WHEEL_TORQUE_LIMIT`. The motor pushes with
  `WHEEL_KD * (asked speed - actual speed)` N*m, capped at the limit.
- **Robot jerks and flips:** lower `WHEEL_TORQUE_LIMIT`, or make the `action_rate` reward weight more negative.
- **It drives straight but slowly:** `TARGET_SPEED` (0.5 m/s) is far more than this robot can do (driving both wheels at
  8 rad/s by hand in `drive.py` moves it about 0.03 m/s), so the reward simply keeps paying for more speed. Lower it to
  about 0.15 if you want it to hold a steady speed, or raise `MAX_WHEEL_SPEED` and the wheel torque to go faster.
- **It wiggles along the line:** raise the `stay_on_line` and `face_forward` weights, lower `START_YAW`.
- **Make it harder:** raise `START_YAW`, or add friction and mass randomization (mjlab's `mjlab.envs.mdp.dr`) so the
  policy copes with the jerkiness of your rod wheels and not one exact friction value.

## What to expect

A run of 4096 robots takes about 3 to 4 minutes for 500 iterations and about 13 minutes for 2000 on a Pax GPU node.
`Mean reward` starts near -0.1 and reaches about +2 after 500 iterations and about +2.6 after 2000.

Results from the movies (10 seconds, starting pointing 0.4 radians off-axis):

| Robot and training | Distance forward | Sideways (y) | Heading |
|---|---|---|---|
| **Sample robot** (the one in `onshape/` until you export your own), 300 iterations | 4.7 m (0.5 m/s, the target speed) | within 0.1 m | within 2 degrees |
| Author's own Onshape robot with rod wheels, 500 iterations | 1.15 m | down to -0.5 m, ends -0.13 m | wanders between -43 and +38 degrees |
| Author's own Onshape robot with rod wheels, 2000 iterations | 1.05 m | stays within about 0.25 m | stays within about 27 degrees |

With the sample robot the mean reward reaches about +9.8 within 150 iterations (about a minute) and the robot drives
straight at the target speed. Your own robot will do better or worse depending on its wheels: the author's rod wheels are
jerky and slippery, so it crawls at about 0.1 m/s (three times faster than the hand-driven plan in `drive.py`) and still
wobbles in heading. For a robot like that, more training helps, and so do the Tuning ideas above (a gentler
`WHEEL_TORQUE_LIMIT`, heavier steering penalties, a lower `TARGET_SPEED`).

Tested on Pax (GPU node `pax050`, mjlab 1.6.0, torch 2.11 for CUDA 12.8, warp 1.14). The cluster's GPU driver is CUDA
12.9, which is why `03_setup_env.sh` installs CUDA 12 builds of PyTorch and Warp. If you see "NVIDIA driver is too old",
that did not happen: run `python3 setup.py clean` to start from scratch.
