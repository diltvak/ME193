"""One command to connect to the Tufts cluster and check that MuJoCo works there.

    python3 setup.py              # everything
    python3 setup.py --skip-gpu   # no GPU job
    python3 setup.py --reinstall  # rebuild ~/mujoco-env on the cluster
    python3 setup.py clean        # back to factory settings (see below), then reinstall from scratch
    python3 setup.py clean --dry-run   # only show what "clean" would delete

"clean" deletes, on the cluster: ~/mujoco-env (the Python environment), the mujoco-hpc folder with every run in it,
~/.cache and ~/Simulations (--keep-simulations to keep it); on your laptop it empties runs/, results/ and logs/. It asks
you to type "factory reset" first, and refuses to run while you have jobs queued or running.

Run it from this folder, on your laptop (off campus: connect the VPN first). Its first step adds a `Host hpc` entry to
~/.ssh/config (if there is not one), so that afterwards `ssh hpc` logs you in to the cluster. It prints
PASS/FAIL for each step and writes everything to logs/setup-<time>.log. If a step fails,
send that log file to whoever is helping you.
"""
import argparse
import os
import re
import shutil
import socket
import sys
from pathlib import Path

from hpc import HERE, Cluster, StepFailed

SCRIPTS = HERE / "scripts"
FILES = ["02_check_cluster.sh", "03_setup_env.sh", "04_test_mujoco.py", "05_job.sbatch", "06_gpu_job.sbatch",
         "08_demo.py", "08_demo.sbatch"]
results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}: {name} {detail}")


def ensure_ssh_alias(c):
    """Add a `Host hpc` entry to ~/.ssh/config (once), so that `ssh hpc` logs you in to the cluster."""
    cfg = Path.home() / ".ssh" / "config"
    old = cfg.read_text() if cfg.exists() else ""
    if re.search(r"^\s*Host\s+(.*\s)?hpc(\s|$)", old, re.M):
        return False
    block = (f"Host hpc\n    HostName {c.host}\n    User {c.user}\n"
             "    ControlMaster auto\n    ControlPath ~/.ssh/sockets/%r@%h\n    ControlPersist 10m\n")
    cfg.parent.mkdir(mode=0o700, exist_ok=True)
    with open(cfg, "a") as f:
        f.write(("\n" if old and not old.endswith("\n") else "") + ("\n" if old else "") + block)
    os.chmod(cfg, 0o600)
    return True


def factory_reset(c, args):
    """Delete everything this project put on the cluster and your laptop. Returns True if it went ahead."""
    if not c.dir.startswith("~/") or c.dir.rstrip("/") == "~":
        raise StepFailed(f"refusing to delete HPC_DIR={c.dir!r}: it must be a folder inside your home, like ~/mujoco-hpc")
    # Cluster side: the Python environment, the folder our scripts and every run live in, and caches.
    cluster = ["~/mujoco-env", c.dir, "~/.cache"] + ([] if args.keep_simulations else ["~/Simulations"])
    # Laptop side: results copied back from the cluster, and the logs (the log of this run is kept).
    local = [HERE / "runs", HERE / "results", HERE / "logs"]

    c.connect()
    _, jobs = c.ssh("squeue -u $USER -h -o '%i %j %T'", check=False, show=False)
    if jobs.strip():
        print(jobs.rstrip())
        raise StepFailed("you have jobs queued or running (listed above). Cancel them first with: scancel -u $USER")

    print("Disk quota now:")
    c.ssh("quota -s 2>/dev/null | tail -n +3 || true", check=False)
    print("\nOn the cluster this will DELETE (size, folder):")
    c.ssh("du -sh " + " ".join(cluster) + " 2>/dev/null | sort -h", check=False)
    print("\nOn this laptop it will empty:  " + ", ".join(str(p.relative_to(HERE)) + "/" for p in local))
    if args.keep_simulations:
        print("\n~/Simulations is kept (--keep-simulations).")
    else:
        print("\nNOTE: ~/Simulations is deleted too. If it holds anything of yours that these scripts did not make,\n"
              "copy it away first, or rerun with --keep-simulations.")
    print("\nThen it reinstalls everything from scratch (the rest of setup.py).")

    if args.dry_run:
        print("\n--dry-run: nothing was deleted.")
        return False
    if not args.yes and input('\nType "factory reset" to go ahead: ').strip().lower() != "factory reset":
        print("Nothing was deleted.")
        return False

    c.say("deleting on the cluster ...")
    c.ssh("rm -rf " + " ".join(cluster))
    c.say("emptying folders on this laptop ...")
    for folder in local:
        if folder.exists():
            for item in folder.iterdir():
                if item == c.logpath:
                    continue
                shutil.rmtree(item) if item.is_dir() else item.unlink()
    print("\nDisk quota after (can take a minute to update):")
    c.ssh("quota -s 2>/dev/null | tail -n +3 || true", check=False)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["clean"],
                    help="clean = back to factory settings: delete the cluster environment, run folders, caches and "
                         "~/Simulations, empty the local runs/logs/results, then reinstall everything")
    ap.add_argument("--dry-run", action="store_true", help="with clean: only show what would be deleted")
    ap.add_argument("--yes", action="store_true", help="with clean: do not ask for confirmation")
    ap.add_argument("--keep-simulations", action="store_true", help="with clean: do not delete ~/Simulations")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--reinstall", action="store_true", help="delete and rebuild ~/mujoco-env on the cluster")
    ap.add_argument("--cpu-partition", default="batch")
    ap.add_argument("--gpu-partition", default="gpu")
    args = ap.parse_args()

    c = Cluster("setup")
    print(f"log file: {c.logpath}\n")
    if args.command == "clean":
        try:
            if not factory_reset(c, args):
                sys.exit(0)
        except StepFailed as e:
            c.say(f"FAILED: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            c.say("interrupted")
            sys.exit(1)
        print("\nClean. Now reinstalling everything.\n")
    try:
        print("0. Shortcut for logging in")
        added = ensure_ssh_alias(c)
        record("ssh alias", True, "added to ~/.ssh/config: from now on `ssh hpc` logs you in to the cluster"
               if added else "(`ssh hpc` is already set up)")

        print("1. Network")
        try:
            socket.getaddrinfo(c.host, 22)
            socket.create_connection((c.host, 22), timeout=5).close()
            record("can reach " + c.host, True)
        except OSError as e:
            record("can reach " + c.host, False, f"({e}) off campus? connect the Tufts VPN")
            c.say("  continuing anyway; the next step is the real test")

        print("2. Log in")
        c.connect()
        _, out = c.ssh("echo $(whoami) on $(hostname)", show=False)
        record("ssh login", True, out.strip())

        print("3. Copy files to the cluster")
        c.push([SCRIPTS / f for f in FILES if (SCRIPTS / f).exists()], c.dir)   # lands flat in ~/mujoco-hpc
        record("copy files", True, f"-> {c.dir}")

        print("4. Survey the cluster (slurm, partitions, python, disk)")
        _, out = c.ssh(f"cd {c.dir} && bash 02_check_cluster.sh", show=True)
        record("slurm available", "MISSING: sbatch" not in out)
        record("pip can reach the internet", "FAIL: cannot reach pypi" not in out)

        print("5. MuJoCo + mjlab environment (~/mujoco-env; the mjlab download is large, several minutes)")
        have_env = False
        if not args.reinstall:
            rc, _ = c.ssh("~/mujoco-env/bin/python -c 'import mujoco, mjlab, imageio_ffmpeg'", check=False, show=False)
            have_env = rc == 0
        if have_env:
            record("environment already installed", True, "(use --reinstall to rebuild)")
        else:
            c.ssh(f"cd {c.dir} && bash 03_setup_env.sh")
            record("environment installed", True)

        print("6. Quick test on the login node")
        _, out = c.ssh(f"cd {c.dir} && source ~/mujoco-env/bin/activate && python 04_test_mujoco.py", check=False)
        record("mujoco physics on login node", "PASS physics" in out)

        print("7. CPU job through Slurm")
        run_job(c, "05_job.sbatch", "mujoco-test", args.cpu_partition, gpu=False)

        if args.skip_gpu:
            print("8. GPU job: skipped")
        else:
            print("8. GPU job through Slurm (can wait in the queue)")
            run_job(c, "06_gpu_job.sbatch", "mujoco-gpu", args.gpu_partition, gpu=True)
    except StepFailed as e:
        record("stopped", False, str(e))
    except KeyboardInterrupt:
        c.say("interrupted")
    finally:
        bad = [r for r in results if not r[1]]
        print("\n==== Summary ====")
        for name, ok, detail in results:
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
        print("\nAll good. Next: python3 runIt.py --gpu scripts/08_demo.py" if not bad else "\nSomething failed.")
        print(f"Full log: {c.logpath}")
    sys.exit(1 if [r for r in results if not r[1]] else 0)


def run_job(c, sbatch_file, prefix, partition, gpu):
    jobid = c.submit(c.dir, sbatch_file, f"--partition={partition}")
    res = c.wait(jobid)
    text = c.read_remote(f"{c.dir}/{prefix}-{jobid}.out")
    print("    " + text.replace("\n", "\n    ").rstrip())
    record("job completed", res["state"] == "COMPLETED", f"({res['state']} on {res['node']})")
    record("physics on compute node", "PASS physics" in text)
    if gpu:
        record("GPU allocated", "GPU 0:" in text)
        record("EGL rendering", "PASS render via egl" in text, "(optional)" if "PASS render" not in text else "")


if __name__ == "__main__":
    main()
