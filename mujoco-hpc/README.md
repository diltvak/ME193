# MuJoCo on the Tufts HPC cluster

Run MuJoCo simulations on the Tufts cluster from your own laptop, and train robots with reinforcement learning:
a few Python commands, no cluster knowledge needed. Everything is logged so that, if something breaks, you can
send one file to get help.

## Where everything is

```
mujoco-hpc/
  README.md            this file
  config.sh            your username and the cluster address (edit once)
  setup.py             install and test everything on the cluster;  python3 setup.py clean  starts over
  runIt.py             run one of your scripts on the cluster and bring the results home
  status.py            see, fetch or cancel your jobs
  cleanup.py           free disk space on the cluster
  hpc.py               code the four commands above share (ssh, rsync, Slurm, logs). You never run it.
  scripts/             the numbered by-hand steps 00-08b, the smoke test, the bouncing-ball demo
  09_onshape_robot/    build a robot in Onshape, drive it, train it to drive straight (has its own README)
  runs/  logs/  results/   created as you work (results from the cluster, logs of every command)
```

Coming back after a while? Connect the Tufts VPN, open a terminal in this folder, and run `python3 setup.py`:
it re-checks the whole chain and tells you what is broken. Then use `runIt.py` and `status.py`.


## Quick start

Run everything on your **laptop**, in a terminal opened **inside this folder** (VSCode: open the
`mujoco-hpc` folder, then Terminal > New Terminal; check with `pwd`). Off campus, connect the Tufts VPN first.

1. Edit `config.sh` once: put your Tufts username in `HPC_USER`.
2. Check the whole setup (about 2 minutes; you type your password / Duo once):
   ```
   python3 setup.py
   ```
   Every step prints PASS or FAIL. All PASS means MuJoCo works on the cluster, on CPU and GPU nodes.
3. Run the demo, a ball that drops and bounces:
   ```
   python3 runIt.py --gpu scripts/08_demo.py
   ```
   It runs on a GPU node, copies the video back to `runs/<name>/bounce.mp4`, and opens it.

Use `python3` on Mac, Linux and WSL. On a Windows PC, use WSL (see [Windows](#windows)).

## The commands

| Command | What it does |
|---|---|
| `python3 setup.py` | Logs in, copies files, installs MuJoCo into `~/mujoco-env` on the cluster, then tests a quick run, a CPU job and a GPU job. `--skip-gpu` skips the GPU job; `--reinstall` rebuilds the environment. |
| `python3 runIt.py [options] yourscript.py [arguments]` | Copies your script to a fresh folder on the cluster, submits it with Slurm, waits, prints its output, and copies everything it wrote back to `runs/<name>/`. Videos and images open automatically. |
| `python3 status.py` | Shows what is queued, running and recently finished, from any terminal at any time. |
| `python3 cleanup.py` | Shows your disk quota on the cluster, deletes old job results and caches, shows the quota again. |

`runIt.py` options (they work before or after the script name; anything else goes to your script):

| Option | Meaning | Default |
|---|---|---|
| `--gpu` | run on a GPU node | CPU node |
| `--time 01:00:00` | maximum run time (the job is stopped after this) | 30 minutes |
| `--mem 8G` | memory | 4G |
| `--cpus 4` | CPU cores | 2 |
| `--with helper.py` | also copy another file next to the script (repeat for more) | none |
| `--partition NAME` | use a different Slurm queue | `batch`, or `gpu` with `--gpu` |
| `--no-open` | do not open videos when done | opens them |

Example: `python3 runIt.py --gpu --time 01:00:00 my_sim.py 10 out.mp4` runs `my_sim.py 10 out.mp4` on a GPU node
for up to an hour. If one of *your* script's arguments has the same name as a `runIt.py` option, put `--` before
your arguments.

`status.py` options:

| Command | Meaning |
|---|---|
| `python3 status.py` | queued / running now, plus jobs from the last 2 days |
| `python3 status.py 6518997` | one job's state and the end of its output |
| `python3 status.py --watch` | refresh every 10 s until Ctrl-C (add a job id to watch one job) |
| `python3 status.py --fetch 6518997` | copy that job's results to `runs/` (if you stopped watching early) |
| `python3 status.py --cancel 6518997` | stop a job |

A job state of `PENDING` with reason `Resources` or `Priority` just means it is waiting for a free node;
GPU jobs can wait a while. Pressing Ctrl-C while `runIt.py` waits stops the *watching*, not the job; use
`status.py` to check on it later.

## The demo: `scripts/08_demo.py`

A ball drops onto the floor and bounces: under 50 lines, with comments. The cluster has no screen, so
instead of opening a window the script renders each frame offscreen and saves an MP4. The structure is the
pattern for any MuJoCo script you run here:

1. `XML`: describe the world (floor, ball, camera, light).
2. `MjModel` / `MjData`: build the model and its state.
3. Loop: `mj_step` advances the physics; `renderer.render()` takes a picture.
4. Write the frames to a video file.

To make your own, copy `scripts/08_demo.py`, change the `XML` (try a bigger ball, a different `solref` for more or
less bounce, a second body) and run it with `runIt.py`. Whatever your script writes (videos, PNG plots, data
files) comes back to `runs/<name>/`. The CPU and GPU runs give the same physics; the GPU only speeds up
rendering (and GPU-based training such as MJX/JAX later). For an interactive window (`mujoco.viewer`) run
MuJoCo on your laptop instead; the cluster is for long or heavy runs.

## Next: a paddle that learns to bounce a ball (mjlab)

`scripts/1105_pingpong.py` drops a ball onto a ping-pong paddle (a thin disc that slides in x, y, z) driven by a
hand-written rule. `scripts/pingpong_task/` teaches the paddle with reinforcement learning instead: it must send the ball
back up to the height it was dropped from (1.5 m). Same workflow as the robot task below (laptop, in this folder):

```
# train (a few minutes on a GPU; checkpoints come back to runs/train-<time>/logs/rsl_rl/ping_pong/<date>/)
python3 runIt.py --gpu --time 00:30:00 --mem 16G --cpus 4 --with scripts/pingpong_task/pingpong_task.py \
  scripts/pingpong_task/train.py Ping-Pong --env.scene.num-envs 4096

# film the trained paddle (prints the top of each bounce)
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 --with scripts/pingpong_task/pingpong_task.py \
  --with runs/train-XXXXXXXX-XXXXXX/logs/rsl_rl/ping_pong/XXXX-XX-XX_XX-XX-XX/model_999.pt \
  scripts/pingpong_task/make_video.py model_999.pt
```
Rewards, ball/paddle sizes and the contact bounciness are constants at the top of `pingpong_task.py`.

### Two paddles passing the ball

`scripts/1112_pingpong_duo.py` is the same scene with a second paddle 1 m away; each paddle can also tilt, so it
can send the ball sideways (hand-written rule: `python3 runIt.py scripts/1112_pingpong_duo.py`).
`scripts/pingpong_duo_task/` teaches one policy to drive both paddles and pass the ball back and forth:

```
# train (3000 iterations; checkpoints come back to runs/train-<time>/logs/rsl_rl/ping_pong_duo/<date>/)
python3 runIt.py --gpu --time 01:00:00 --mem 16G --cpus 4 --with scripts/pingpong_duo_task/pingpong_duo_task.py \
  scripts/pingpong_duo_task/train.py Ping-Pong-Duo --env.scene.num-envs 4096

# film the trained paddles (prints each completed pass)
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 --with scripts/pingpong_duo_task/pingpong_duo_task.py \
  --with runs/train-XXXXXXXX-XXXXXX/logs/rsl_rl/ping_pong_duo/XXXX-XX-XX_XX-XX-XX/model_2999.pt \
  scripts/pingpong_duo_task/make_video.py model_2999.pt
```

## Next: your own robot from Onshape

`09_onshape_robot/` has a second demo: design a box-with-two-wheels robot in Onshape, export it to MuJoCo
automatically, and drive it. Start with `09_onshape_robot/README.md`.

## If something goes wrong

Every run writes a log of every command and its output to `logs/`. **Send the newest file in `logs/`**
(the last line of each command's output tells you its name).

| What you see | What it means / what to do |
|---|---|
| `STOP: this script is for your laptop` | You are in a terminal logged in to the cluster (prompt says `login-...`). Use the terminal on your own computer. |
| `can reach ...` FAIL, or login fails | Off campus without the Tufts VPN, or wrong `HPC_USER` / `HPC_HOST` in `config.sh`. |
| Asked for a password again | The shared login lasts 2 hours. Just type it again. |
| `no such file` when running a script | The terminal is not in the `mujoco-hpc` folder. `cd` there. (Scripts are called `scripts/08_demo.py`, not `08_demo.py`.) |
| `the MuJoCo environment is missing` | Run `python3 setup.py` first. |
| Job stays `PENDING` | Waiting for a free node (normal for GPUs). `python3 status.py` shows the reason. |
| Job `FAILED` | Read the job output printed by `runIt.py` (or `python3 status.py <JOBID>`); the traceback is usually in the last lines. |
| `TIMEOUT` | The job hit its time limit. Raise `--time`. |
| `PartitionNotFound` / `InvalidQOS` | Wrong queue name. Use `--partition` with a name from `sinfo -s` on the cluster. |
| Quota warnings on the cluster | Home folders are limited (about 28 GB). Run `python3 cleanup.py` to delete old runs and caches. |
| `NVIDIA driver ... too old` when training | The wrong (CUDA 13) PyTorch is installed. Run `python3 setup.py clean` to start from scratch. |
| `libEGL warning` / `No 3D enabled` lines | Harmless noise from rendering without a display. The smoke test now hides them; if you still see them from your own script, ignore them as long as it works. |

### Back to factory settings

If the cluster setup is broken beyond repair, or you want to begin again:
```
python3 setup.py clean --dry-run   # only shows what would be deleted
python3 setup.py clean
```
`clean` deletes, **on the cluster**, the Python environment `~/mujoco-env`, the `~/mujoco-hpc` folder with every run in
it, `~/.cache`, and `~/Simulations`; and it empties `runs/`, `results/` and `logs/` **on your laptop**. It shows the sizes,
and you must type `factory reset` before anything is deleted. It will not run while you have jobs queued or running
(cancel them with `scancel -u $USER`). Add `--keep-simulations` if `~/Simulations` holds anything of yours. When it
has deleted everything it runs the whole setup again from scratch. Files outside those folders are never touched.

## Windows

Windows has no `rsync`, and its built-in `ssh` cannot share one login between commands (you would be asked
for Duo on every step). So use WSL (Ubuntu inside Windows), about 10 minutes once:

1. PowerShell as administrator: `wsl --install`, restart, finish the Ubuntu setup.
2. In the Ubuntu app: `sudo apt update && sudo apt install -y rsync openssh-client python3`
3. Keep this folder inside Ubuntu (for example `~/mujoco-hpc`, not under `/mnt/c`) and run the same commands
   with `python3`. In VSCode, install the "WSL" extension and choose "Connect to WSL".

Running the scripts in plain Windows stops with these instructions.

## What is in this folder

Every file, and what it is for. *Laptop* = runs in a terminal on your computer, *cluster* = runs on the cluster.

| File | Runs on | What it does |
|---|---|---|
| `config.sh` | laptop | Your username (`HPC_USER`), the cluster address and the folder used on the cluster. The only thing you edit. |
| `setup.py` | laptop | Logs in, adds the `ssh hpc` shortcut, copies the `scripts/` files, builds the Python environment `~/mujoco-env` on the cluster, and tests a login-node run, a CPU job and a GPU job. `clean` deletes everything and reinstalls. |
| `runIt.py` | laptop | Runs one of your scripts on the cluster (see above) and copies the results to `runs/<name>/`. |
| `status.py` | laptop | Lists, watches, fetches and cancels your Slurm jobs. |
| `cleanup.py` | laptop | Quota check, then deletes old runs and caches on the cluster. |
| `hpc.py` | (library) | Shared code: reads `config.sh`, opens one SSH login that every command reuses, copies files with rsync, submits and waits for Slurm jobs, writes the logs. The four commands above import it; you never run it. |
| `scripts/` | | The pieces `setup.py` uses, and the by-hand route. Table below. |
| `09_onshape_robot/` | | Second demo: build a robot in Onshape, export it, drive it. `mjlab_task/` inside trains it to drive straight with reinforcement learning. Has its own README. |
| `runs/` | laptop | One folder per `runIt.py` job: your script, its output, videos, checkpoints. Safe to delete. |
| `logs/` | laptop | A log of every command and its output, one file per run. Send the newest one when asking for help. |
| `results/` | laptop | Videos fetched by `scripts/08b_fetch_results.sh` (by-hand route only). |

### The `scripts/` folder

The numbers are the order of the by-hand steps (there is no 07). `setup.py` copies the files marked *auto* to the cluster
and runs them for you.

| File | Runs on | What it does |
|---|---|---|
| `00_check_connection.sh` | laptop | Is the cluster reachable? Checks the VPN, DNS, port 22 and the SSH login. |
| `01_push_and_check.sh` | laptop | Copies the scripts to the cluster and runs `02_check_cluster.sh` there. |
| `01b_login.sh` | laptop | Opens a terminal on the cluster in `~/mujoco-hpc` (like `ssh hpc`). |
| `02_check_cluster.sh` | cluster, *auto* | Read-only survey: Slurm, queues, Python, modules, disk, pip access. |
| `03_setup_env.sh` | cluster, *auto* | Builds `~/mujoco-env`: a Python 3.11+ environment with CUDA 12 PyTorch and Warp, mjlab, MuJoCo, numpy and imageio. About 7 GB. |
| `04_test_mujoco.py` | cluster, *auto* | Smoke test: drops a box and checks it rests on the floor; tries offscreen rendering. |
| `05_job.sbatch` | cluster, *auto* | Slurm job that runs the smoke test on a CPU compute node. |
| `06_gpu_job.sbatch` | cluster, *auto* | The same on a GPU node, with GPU rendering. |
| `08_demo.py` | cluster | The bouncing-ball demo (use it with `runIt.py`, and copy it to start your own). |
| `08_demo.sbatch` | cluster | Slurm job for the demo, for the by-hand route. |
| `08b_fetch_results.sh` | laptop | Copies the demo video from the cluster to `results/` and opens it. |
| `activate.sh` | cluster | `source activate.sh` in a new cluster terminal: loads Python and switches to `~/mujoco-env`. |

On the cluster these files sit side by side in `~/mujoco-hpc` (no `scripts/` folder there), so the commands you type
on the cluster have no `scripts/` in front.

What the setup found on the cluster (Tufts "Pax"): login node `login.pax.tufts.edu`; Slurm queues `batch`
(CPU) and `gpu` (NVIDIA A100 nodes); Python 3.13 via `module load python/3.13.2` (the system Python 3.9 is too old
for MuJoCo, so the installer loads the newer one automatically); no preinstalled MuJoCo, so each user
gets their own in `~/mujoco-env`. That environment includes mjlab and PyTorch and takes about 6 GB of your
home quota (28 GB on Pax). Check it with `quota -s`, and free space with `python3 cleanup.py`.

## Logging in to the cluster yourself

`python3 setup.py` adds a shortcut to your `~/.ssh/config` the first time it runs, so from then on this opens a terminal
on the cluster (same login as the scripts, so no new password prompt while their connection is open):
```
ssh hpc
```
Type `exit` to come back to your laptop. Useful for `quota -s` (your disk space) and `squeue -u $USER` (your jobs).

## Doing it by hand (optional)

The Python commands do all of this for you. The by-hand steps are for understanding what happens, or debugging.
Scripts marked *laptop* run in a terminal on your computer; *cluster* in a terminal logged in to the cluster.

| Step | Where | Command | Proves |
|---|---|---|---|
| 0 | laptop | edit `config.sh` | username/host are right |
| 1 | laptop | `./scripts/00_check_connection.sh` | VPN, DNS, SSH, Duo all work |
| 2 | laptop | `./scripts/01_push_and_check.sh` | files copy over; runs the survey (step 3) too |
| 2.5 | laptop, 2nd terminal | `./scripts/01b_login.sh` | you are logged in, in `~/mujoco-hpc` (VSCode: **+** in the terminal panel) |
| 3 | cluster | `bash 02_check_cluster.sh` | Slurm, queues, Python, modules, disk, pip reachable |
| 4 | cluster | `bash 03_setup_env.sh` | MuJoCo, mjlab (with PyTorch), numpy, imageio, imageio-ffmpeg install into `~/mujoco-env` |
| 5 | cluster | `source ~/mujoco-env/bin/activate`, then `python 04_test_mujoco.py` | MuJoCo imports and simulates |
| 6 | cluster | `sbatch 05_job.sbatch` | same test on a compute node via Slurm |
| 7 | cluster | `sbatch 06_gpu_job.sbatch` | a GPU node is allocated and EGL rendering works |
| 8 | cluster, then **laptop** | `sbatch 08_demo.sbatch`, then `./scripts/08b_fetch_results.sh` | the demo renders; the video comes home and opens |

Every new cluster terminal needs `source ~/mujoco-env/bin/activate` first (if `python --version` says 3.9, you
skipped it). Run the laptop scripts (`./scripts/...`) from inside the `mujoco-hpc` folder. On the cluster the numbered files are in `~/mujoco-hpc` itself, so there you type `bash 02_check_cluster.sh`, not `scripts/...`.

Checking a job by hand, on the cluster:
```
squeue -u $USER                  # listed = waiting (PD) or running (R); gone = finished
sacct -j <jobid> -X --format=JobID,State,Elapsed,ExitCode   # COMPLETED = ok; FAILED / TIMEOUT: read the .out file
tail -f mujoco-gpu-<jobid>.out   # watch output live (Ctrl-C stops watching, not the job)
cat mujoco-gpu-<jobid>.out       # the result, once done
```
`sbatch` prints `Submitted batch job <jobid>`; that number is in the output file name. Don't run long
simulations on the login node; submit them with `sbatch` (or `runIt.py`).
