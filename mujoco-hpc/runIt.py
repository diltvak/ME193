"""Run one of your MuJoCo scripts on the cluster and bring the results home.

    python3 runIt.py scripts/08_demo.py                  # CPU node
    python3 runIt.py --gpu scripts/08_demo.py 6 out.mp4  # GPU node; "6 out.mp4" are passed to the script
    python3 runIt.py scripts/08_demo.py --gpu            # same thing: options work in either place
    python3 runIt.py --time 02:00:00 --mem 8G --gpu my_sim.py

runIt's options (--gpu, --time, ...) can go before or after the script name; anything else is
handed to your script.
It copies the script to a fresh folder on the cluster, submits it with Slurm, waits, and copies
everything the script wrote back to ./runs/<name>/ (videos open automatically).
The log is in logs/runIt-<time>.log.
"""
import argparse
import datetime
import shlex
import sys
import tempfile
from pathlib import Path

from hpc import HERE, Cluster, StepFailed, open_file

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
{gres}#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
PYMOD=$(cat "$HOME/mujoco-env/python_module" 2>/dev/null)
[ -n "$PYMOD" ] && module load "$PYMOD"
source "$HOME/mujoco-env/bin/activate"
export MUJOCO_GL=egl
python -u {script} {args}   # -u: write output as it happens, so status.py can show a job's progress
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--gpu", action="store_true", help="run on a GPU node")
    ap.add_argument("--partition", help="override the Slurm partition (default batch, or gpu with --gpu)")
    ap.add_argument("--time", default="00:30:00")
    ap.add_argument("--mem", default="4G")
    ap.add_argument("--cpus", default="2")
    ap.add_argument("--with", dest="extra", action="append", default=[], help="extra file to copy next to the script (repeatable)")
    ap.add_argument("--no-open", action="store_true", help="do not open videos when done")
    ap.add_argument("script")
    # runIt's own options work before or after the script name; everything else goes to your script.
    a, a.args = ap.parse_known_args()
    if a.args[:1] == ["--"]:
        a.args = a.args[1:]

    script = Path(a.script)
    if not script.exists():
        sys.exit(f"no such file: {script}")
    name = f"{script.stem}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    rdir = f"{{dir}}/runs/{name}"

    c = Cluster("runIt")
    rdir = rdir.format(dir=c.dir)
    print(f"log file: {c.logpath}\n")
    ok = False
    try:
        c.connect()
        rc, _ = c.ssh("test -x ~/mujoco-env/bin/python", check=False, show=False)
        if rc == 255:   # ssh itself failed, so this says nothing about the environment
            raise StepFailed("cannot reach the cluster: no network or VPN, or the shared login has expired "
                             "(it lasts 2 hours). Reconnect with: python3 setup.py   (or ./01b_login.sh), then retry.")
        if rc != 0:
            raise StepFailed("the MuJoCo environment is missing on the cluster. Run: python3 setup.py")

        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job.sbatch"
            job.write_text(TEMPLATE.format(
                name=script.stem[:20], partition=a.partition or ("gpu" if a.gpu else "batch"),
                gres="#SBATCH --gres=gpu:1\n" if a.gpu else "", time=a.time, cpus=a.cpus, mem=a.mem,
                script=shlex.quote(script.name), args=" ".join(shlex.quote(x) for x in a.args)))
            c.say(f"copying to {rdir}")
            c.push([script, job, *[Path(x) for x in a.extra]], rdir)

        jobid = c.submit(rdir, "job.sbatch")
        res = c.wait(jobid)
        print("\n---- job output ----")
        print(c.read_remote(f"{rdir}/slurm-{jobid}.out").rstrip())
        print("--------------------")

        local = HERE / "runs" / name
        c.pull(rdir, local)
        files = sorted(p for p in local.iterdir() if p.name not in ("job.sbatch", script.name))
        c.say(f"copied back to {local}:")
        for p in files:
            c.say(f"  {p.name}  ({p.stat().st_size / 1e3:.0f} kB)")
        ok = res["state"] == "COMPLETED"
        if not a.no_open:
            for p in files:
                if p.suffix in (".mp4", ".png", ".gif"):
                    open_file(p)
    except StepFailed as e:
        c.say(f"FAILED: {e}")
    except KeyboardInterrupt:
        c.say("interrupted")
    print("\n" + ("Done." if ok else "Something failed; see the output above."))
    print(f"Full log: {c.logpath}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
